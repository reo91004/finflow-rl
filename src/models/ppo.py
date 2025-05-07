"""
Proximal Policy Optimization(PPO) 알고리즘 모듈

주식 포트폴리오 관리를 위한 PPO 알고리즘을 구현합니다.
Actor-Critic 모델을 사용하여 학습하며, EMA(Exponential Moving Average) 모델,
Early stopping, 학습률 스케줄러 등 다양한 안정화 기법을 적용합니다.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import traceback
import gc
from src.models.actor_critic import ActorCritic
from src.constants import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LR,
    DEFAULT_GAMMA,
    DEFAULT_K_EPOCHS,
    DEFAULT_EPS_CLIP,
    MODEL_SAVE_PATH,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    VALIDATION_INTERVAL,
    VALIDATION_EPISODES,
    LR_SCHEDULER_T_MAX,
    LR_SCHEDULER_ETA_MIN,
    LAMBDA_GAE,
    RMS_EPSILON,
    CLIP_OBS,
    BATCH_SIZE,
    GRADIENT_CLIP,
    ENTROPY_COEF,
    CRITIC_COEF
)

class PPO:
    """
    Proximal Policy Optimization (PPO) 알고리즘 클래스입니다.
    Actor-Critic 모델을 사용하여 포트폴리오 관리 문제를 학습합니다.
    """

    def __init__(
        self,
        n_assets,
        n_features,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lr=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        k_epochs=DEFAULT_K_EPOCHS,
        eps_clip=DEFAULT_EPS_CLIP,
        model_path=MODEL_SAVE_PATH,
        logger=None,
        use_ema=True,
        ema_decay=0.995,  # 0.99 → 0.995
        use_lr_scheduler=True,
        use_early_stopping=True,
    ):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.model_path = model_path
        self.logger = logger or logging.getLogger("PortfolioRL")  # 로거 없으면 기본 설정 사용
        self.n_assets = n_assets
        self.n_features = n_features  # 추가

        # EMA 가중치 옵션
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 학습률 스케줄러 및 Early Stopping 설정
        self.use_lr_scheduler = use_lr_scheduler
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.best_validation_reward = -float("inf")
        self.no_improvement_episodes = 0
        self.should_stop_early = False

        os.makedirs(model_path, exist_ok=True)

        # 정책 네트워크 (현재 정책, 이전 정책)
        self.policy = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 가중치 복사

        # EMA 모델 (학습 안정성을 위한 Exponential Moving Average)
        if self.use_ema:
            self.policy_ema = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
            self.policy_ema.load_state_dict(self.policy.state_dict())
            # EMA 모델의 파라미터는 업데이트되지 않도록 설정
            for param in self.policy_ema.parameters():
                param.requires_grad = False

        # 옵티마이저 및 손실 함수
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()  # 크리틱 손실용

        # 학습률 스케줄러 (Cosine Annealing)
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=LR_SCHEDULER_T_MAX, eta_min=LR_SCHEDULER_ETA_MIN
            )
            self.logger.info(
                f"Cosine Annealing LR 스케줄러 설정: T_max={LR_SCHEDULER_T_MAX}, eta_min={LR_SCHEDULER_ETA_MIN}"
            )

        self.best_reward = -float("inf")  # 최고 성능 모델 저장을 위한 변수
        self.obs_rms = None  # 학습된 상태 정규화 통계 저장용

        # GPU 설정 (성능 향상 최적화 옵션)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 행렬 곱셈 연산 정밀도 설정 (A100/H100 등 TensorFloat32 지원 시 유리)
            # torch.set_float32_matmul_precision('high') # 또는 'medium'
            
        # 배치 정규화 변수 추가
        self.use_batch_norm = True

    def update_lr_scheduler(self):
        """학습률 스케줄러를 업데이트합니다."""
        if self.use_lr_scheduler and self.scheduler:
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            return current_lr
        return None

    def validate(self, env, n_episodes=VALIDATION_EPISODES):
        """
        현재 정책을 검증하여 Early Stopping에 사용할 보상을 계산합니다.

        Args:
            env: 검증에 사용할 환경 (StockPortfolioEnv)
            n_episodes: 실행할 검증 에피소드 수

        Returns:
            float: 평균 검증 보상
        """
        # 평가 모드로 설정
        self.policy_old.eval()
        if self.use_ema:
            self.policy_ema.eval()

        total_reward = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # EMA 모델 사용 (있는 경우)
                if self.use_ema:
                    with torch.no_grad():
                        state_tensor = (
                            torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                        )
                        action_probs, _ = self.policy_ema(state_tensor)
                        dist = torch.distributions.Categorical(action_probs)
                        action_idx = dist.sample()

                        # 원-핫 인코딩으로 변환
                        action = torch.zeros_like(action_probs)
                        action.scatter_(1, action_idx.unsqueeze(-1), 1.0)
                        action = action.squeeze(0).cpu().numpy()
                else:
                    action, _, _ = self.policy_old.act(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += info.get("raw_reward", reward)

                if terminated or truncated:
                    done = True
                else:
                    state = next_state

            total_reward += episode_reward

        # 학습 모드로 복원
        self.policy_old.train()
        if self.use_ema:
            self.policy_ema.train()

        # 평균 검증 보상 반환
        return total_reward / n_episodes

    def check_early_stopping(self, validation_reward):
        """
        검증 보상에 기반하여 Early Stopping 여부를 확인합니다.

        Args:
            validation_reward: 현재 검증 보상

        Returns:
            bool: True면 학습 중단, False면 계속 진행
        """
        if not self.use_early_stopping:
            return False

        if validation_reward > self.best_validation_reward:
            # 성능 향상이 있으면 최고 기록 갱신 및 인내심 카운터 리셋
            self.best_validation_reward = validation_reward
            self.no_improvement_episodes = 0
            return False
        else:
            # 성능 향상이 없으면 인내심 카운터 증가
            self.no_improvement_episodes += 1

            # 로깅
            self.logger.info(
                f"최고 검증 보상 {self.best_validation_reward:.4f} 대비 향상 없음. "
                f"인내심 카운터: {self.no_improvement_episodes}/{self.early_stopping_patience}"
            )

            # 인내심 카운터가 임계값을 넘으면 학습 중단
            if self.no_improvement_episodes >= self.early_stopping_patience:
                self.logger.warning(
                    f"Early Stopping 조건 충족! {self.early_stopping_patience} 에피소드 동안 "
                    f"성능 향상 없음. 최고 검증 보상: {self.best_validation_reward:.4f}"
                )
                self.should_stop_early = True
                return True

        return False

    def update_ema_model(self):
        """
        EMA(Exponential Moving Average) 모델의 가중치를 업데이트합니다.
        ema_weight = decay * ema_weight + (1 - decay) * current_weight
        """
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, current_param in zip(
                self.policy_ema.parameters(), self.policy.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    current_param.data, alpha=1.0 - self.ema_decay
                )

    def save_model(self, episode, reward, save_file=None):
        """
        모델의 가중치와 옵티마이저 상태, obs_rms 통계를 저장합니다.
        
        Args:
            episode (int): 현재 에피소드 번호
            reward (float): 현재 보상 
            save_file (str, optional): 저장할 파일 경로. None이면 best_model.pth 또는 final_model.pth로 저장
        
        Returns:
            bool: 저장 성공 여부
        """
        # 저장 경로 설정
        if save_file is None:
            if reward > self.best_reward:
                self.best_reward = reward
                save_file = os.path.join(self.model_path, "best_model.pth")
            else:
                save_file = os.path.join(self.model_path, "final_model.pth")
        else:
            # 외부에서 경로 지정하는 경우 (체크포인트)
            if reward > self.best_reward:
                self.best_reward = reward
            
        try:
            # 저장할 데이터 구성
            checkpoint = {
                "episode": episode,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_reward": self.best_reward,
            }

            # EMA 모델이 있으면 함께 저장
            if self.use_ema:
                checkpoint["ema_model_state_dict"] = self.policy_ema.state_dict()

            # obs_rms 있으면 함께 저장
            if self.obs_rms is not None:
                checkpoint["obs_rms"] = self.obs_rms

            # 파일 저장
            torch.save(checkpoint, save_file)
            self.logger.info(f"모델 저장 완료: {save_file} (보상: {reward:.4f})")
            return True
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def load_model(self, model_file=None):
        """저장된 모델 파일에서 가중치와 옵티마이저 상태를 불러옵니다."""
        if model_file is None:
            model_file = os.path.join(self.model_path, "best_model.pth")

        if not os.path.isfile(model_file):
            self.logger.warning(f"모델 파일을 찾을 수 없음: {model_file}")
            return False

        try:
            # 장치 확인 (CPU 또는 GPU)
            map_location = torch.device("cpu") if not torch.cuda.is_available() else None
            checkpoint = torch.load(model_file, map_location=map_location)

            # 모델 가중치 불러오기
            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.policy_old.load_state_dict(checkpoint["model_state_dict"])

            # EMA 모델 불러오기 (있는 경우)
            if self.use_ema and "ema_model_state_dict" in checkpoint:
                self.policy_ema.load_state_dict(checkpoint["ema_model_state_dict"])
            elif self.use_ema:
                # EMA 모델 가중치가 없으면 현재 정책으로 초기화
                self.policy_ema.load_state_dict(self.policy.state_dict())

            # 옵티마이저 상태 불러오기 (있는 경우)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # 관측 정규화 통계 불러오기 (있는 경우)
            if "obs_rms" in checkpoint:
                self.obs_rms = checkpoint["obs_rms"]

            self.best_reward = checkpoint.get("best_reward", self.best_reward)
            episode = checkpoint.get("episode", 0)

            self.logger.info(
                f"모델 불러오기 완료: {model_file} "
                f"(에피소드: {episode}, 최고 보상: {self.best_reward:.4f})"
            )
            return True
        except Exception as e:
            self.logger.error(f"모델 불러오기 중 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def select_action(self, state, use_ema=True):
        """
        현재 상태에서 행동을 선택합니다.
        주로 테스트 또는 시뮬레이션 중에 사용됩니다.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            # EMA 모델이 있고 사용 설정이 켜져 있으면 EMA 모델 사용
            if self.use_ema and use_ema:
                action_probs, _ = self.policy_ema(state_tensor)
            else:
                action_probs, _ = self.policy_old(state_tensor)

            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()

            # 원-핫 인코딩으로 변환
            action = torch.zeros_like(action_probs)
            action.scatter_(1, action_idx.unsqueeze(-1), 1.0)

        return action.squeeze(0).cpu().numpy()

    def compute_returns_and_advantages(self, rewards, is_terminals, values):
        """
        Generalized Advantage Estimation (GAE)를 사용하여 Advantage와 Return을 계산합니다.

        Args:
            rewards (list): 에피소드/배치에서 얻은 보상 리스트.
            is_terminals (list): 각 스텝의 종료 여부 리스트.
            values (np.ndarray): 각 상태에 대한 크리틱의 가치 예측값 배열.

        Returns:
            tuple: (returns_tensor, advantages_tensor)
                   - returns_tensor (torch.Tensor): 계산된 Return (Target Value).
                   - advantages_tensor (torch.Tensor): 계산된 Advantage.
                   오류 발생 시 빈 텐서 반환.
        """
        if not rewards or values.size == 0:
            self.logger.warning("GAE 계산 시 rewards 또는 values 배열이 비어있습니다.")
            return torch.tensor([], device=DEVICE), torch.tensor([], device=DEVICE)

        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0.0

        next_value = values[-1] * (1.0 - float(is_terminals[-1]))

        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(is_terminals[i])
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            last_gae_lam = delta + self.gamma * LAMBDA_GAE * mask * last_gae_lam
            advantages[i] = last_gae_lam
            returns[i] = last_gae_lam + values[i]
            next_value = values[i]

        try:
            returns_tensor = torch.from_numpy(returns).float().to(DEVICE)
            advantages_tensor = torch.from_numpy(advantages).float().to(DEVICE)
        except Exception as e:
            self.logger.error(f"Return/Advantage 텐서 변환 중 오류: {e}")
            return torch.tensor([], device=DEVICE), torch.tensor([], device=DEVICE)

        if torch.isnan(returns_tensor).any() or torch.isinf(returns_tensor).any():
            returns_tensor = torch.nan_to_num(returns_tensor, nan=0.0)
        if torch.isnan(advantages_tensor).any() or torch.isinf(advantages_tensor).any():
            advantages_tensor = torch.nan_to_num(advantages_tensor, nan=0.0)

        return returns_tensor, advantages_tensor

    def update(self, memory):
        """ 메모리에 저장된 경험을 사용하여 정책(policy)을 업데이트합니다. """
        if not memory.states:
            self.logger.warning("업데이트 시도: 메모리가 비어있습니다.")
            return 0.0

        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        try:
            # 1. 메모리에서 데이터 로드
            old_states = torch.stack([torch.from_numpy(s).float() for s in memory.states]).to(DEVICE)
            old_actions = torch.stack([torch.from_numpy(a).float() for a in memory.actions]).to(DEVICE)
            old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(DEVICE)
            old_values = torch.tensor(memory.values, dtype=torch.float32).to(DEVICE)
            
            # 원시 보상 사용 (메모리 재사용 문제 해결)
            rewards = memory.get_raw_rewards_tensor().to(DEVICE)
            
            # 2. GAE 계산
            old_values_np = old_values.cpu().numpy()
            returns, advantages = self.compute_returns_and_advantages(
                memory.raw_rewards, memory.is_terminals, old_values_np
            )

            if returns.numel() == 0 or advantages.numel() == 0:
                self.logger.error("GAE 계산 실패로 PPO 업데이트 중단.")
                return 0.0

            # 3. Advantage 정규화
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                self.logger.warning("Advantage 정규화 후 NaN/Inf 발견. 0으로 대체.")
                advantages = torch.nan_to_num(advantages, nan=0.0)

            # 4. 미니배치 학습 (메모리 크기에 따라 배치 수 조정)
            batch_size = min(BATCH_SIZE, len(memory.states))
            n_batches = len(memory.states) // batch_size
            
            # 각 에포크별 손실 추적
            epoch_losses = []
            
            for _ in range(self.k_epochs):
                # 데이터 인덱스를 셔플하여 다양한 배치 구성
                indices = torch.randperm(len(memory.states))
                epoch_loss = 0.0
                epoch_actor_loss = 0.0
                epoch_critic_loss = 0.0
                epoch_entropy = 0.0
                
                # 미니배치 단위로 처리
                for i in range(n_batches):
                    batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                    
                    # 미니배치 데이터 추출
                    states_batch = old_states[batch_indices]
                    actions_batch = old_actions[batch_indices]
                    logprobs_batch = old_logprobs[batch_indices]
                    advantages_batch = advantages[batch_indices]
                    returns_batch = returns[batch_indices]
                    
                    # 현재 정책으로 배치 평가
                    batch_logprobs, batch_entropy, batch_state_values = self.policy.evaluate(
                        states_batch, actions_batch
                    )
                    
                    # PPO 비율 계산
                    ratios = torch.exp(batch_logprobs - logprobs_batch.detach())
                    
                    # PPO 손실 계산 (클리핑 적용)
                    surr1 = ratios * advantages_batch
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # 가치 손실 계산 (Huber Loss 또는 MSE)
                    # state_values와 returns의 shape 맞추기
                    batch_state_values = batch_state_values.reshape(-1)  # [batch_size, 1] -> [batch_size]
                    
                    # Huber Loss와 MSE 손실 결합 - 이상치에 더 강건함
                    critic_loss_mse = self.MseLoss(batch_state_values, returns_batch)
                    critic_loss_huber = F.smooth_l1_loss(batch_state_values, returns_batch)
                    critic_loss = 0.5 * critic_loss_mse + 0.5 * critic_loss_huber
                    
                    # 엔트로피 보너스 (탐색 촉진)
                    entropy_loss = batch_entropy.mean()
                    
                    # 최종 손실 계산
                    loss = actor_loss + CRITIC_COEF * critic_loss - ENTROPY_COEF * entropy_loss
                    
                    # 손실 체크 (NaN/Inf 방지)
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"배치 {i}에서 NaN/Inf 손실 발생. 해당 배치 건너뜁니다.")
                        continue
                    
                    # 옵티마이저 업데이트
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 그래디언트 클리핑 (안정성 증가)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=GRADIENT_CLIP)
                    
                    self.optimizer.step()
                    
                    # 손실 누적
                    epoch_loss += loss.item()
                    epoch_actor_loss += actor_loss.item()
                    epoch_critic_loss += critic_loss.item()
                    epoch_entropy += entropy_loss.item()
                    
                # 에포크별 평균 손실 계산 (유효 배치 수로 나눔)
                if n_batches > 0:
                    epoch_losses.append(epoch_loss / n_batches)
                    total_actor_loss += epoch_actor_loss / n_batches
                    total_critic_loss += epoch_critic_loss / n_batches
                    total_entropy += epoch_entropy / n_batches
                    
                # 온도 파라미터 업데이트
                self.policy.update_temperature()
                
                # EMA 모델 가중치 업데이트
                if self.use_ema:
                    self.update_ema_model()
                
                # KL 발산 조기 종료 - 정책이 너무 크게 변경되는 것 방지
                if len(epoch_losses) >= 2 and epoch_losses[-1] > 1.5 * epoch_losses[-2]:
                    self.logger.debug(f"KL 발산 탐지: 에폭 {_+1}/{self.k_epochs}에서 업데이트 조기 종료")
                    break
            
            # 이전 정책 업데이트
            self.policy_old.load_state_dict(self.policy.state_dict())
            
            # 로깅을 위한 손실 평균값 계산
            if epoch_losses:
                total_loss = sum(epoch_losses) / len(epoch_losses)
                self.logger.debug(f"PPO 업데이트 완료: 손실={total_loss:.4f}, 액터={total_actor_loss/self.k_epochs:.4f}, 크리틱={total_critic_loss/self.k_epochs:.4f}, 엔트로피={total_entropy/self.k_epochs:.4f}")
                
            return total_loss

        except Exception as e:
            self.logger.error(f"PPO 업데이트 중 예상치 못한 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return 0.0 