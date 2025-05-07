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
        GAE(Generalized Advantage Estimation)를 사용하여 Returns와 Advantages를 계산합니다.

        Args:
            rewards: 보상 배열 [T]
            is_terminals: 종료 상태 플래그 배열 [T]
            values: 가치 함수 추정치 배열 [T]

        Returns:
            returns: 기대 수익 배열 [T]
            advantages: 어드밴티지 배열 [T]
        """
        T = len(rewards)
        
        # 다음 상태 가치 추정 (마지막 상태는 종료 상태이거나 최대 길이를 초과한 경우 0으로 설정)
        next_values = torch.cat([values[1:], torch.zeros(1).to(DEVICE)])
        
        # 마스크 생성 (종료 상태는 0, 그렇지 않으면 1)
        masks = 1.0 - is_terminals.float()
        
        # GAE 계산
        gae = 0
        returns = []
        advantages = []
        
        for t in reversed(range(T)):
            # 델타 값 계산: rt + gamma * V(s_{t+1}) * mask - V(s_t)
            delta = rewards[t] + self.gamma * next_values[t] * masks[t] - values[t]
            
            # GAE 재귀적 계산: A_t = delta_t + gamma * lambda * A_{t+1} * mask
            gae = delta + self.gamma * LAMBDA_GAE * masks[t] * gae
            
            # returns와 advantages 앞에 추가 (역순 계산)
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        # 텐서로 변환
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        
        # NaN/Inf 처리 (안정성 강화)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=5.0, neginf=-5.0)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # advantages 정규화 (학습 안정화)
        if advantages.numel() > 1:  # 단일 요소가 아닌 경우에만 정규화
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def update(self, memory):
        """
        수집된 경험으로부터 PPO 정책 업데이트를 수행합니다.

        Args:
            memory: 수집된 경험이 저장된 메모리 객체

        Returns:
            float: 정책 손실 값
        """
        try:
            # 메모리가 충분히 차 있는지 확인
            if len(memory) < 100:  # 최소 샘플 수
                self.logger.warning(f"메모리에 충분한 샘플이 없음: {len(memory)}개. 업데이트 건너뜀")
                return 0

            # 메모리에서 배치 데이터 가져오기
            states = memory.get_states_tensor().to(DEVICE)
            actions = memory.get_actions_tensor().to(DEVICE)
            old_log_probs = memory.get_log_probs_tensor().to(DEVICE)
            rewards = memory.get_rewards_tensor().to(DEVICE)
            is_terminals = memory.get_is_terminals_tensor().to(DEVICE)
            
            # 이상치 처리: 보상 범위가 이미 환경에서 클리핑되었으므로 여기서는 극단적인 값만 처리
            # 환경에서의 클리핑 범위보다 약간 더 넓게 설정 (-3.0~3.0)
            rewards = torch.clamp(rewards, -3.0, 3.0)
            
            # 현재 가치 추정
            _, old_values = self.policy_old(states)
            old_values = old_values.detach().squeeze()
            
            # GAE를 사용하여 returns와 advantages 계산
            returns, advantages = self.compute_returns_and_advantages(
                rewards, is_terminals, old_values
            )

            # 미니배치 학습
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            # 데이터셋 크기
            dataset_size = len(states)
            
            # 미니배치 크기 설정
            minibatch_size = min(BATCH_SIZE, dataset_size)
            
            # 미니배치 수 계산
            n_minibatches = max(1, dataset_size // minibatch_size)
            
            # 학습 안정화를 위한 손실 임계값
            max_value_loss = 2.0
            max_policy_loss = 1.0
            
            # k_epochs만큼 반복 업데이트
            for epoch in range(self.k_epochs):
                # 데이터셋 인덱스 셔플
                indices = torch.randperm(dataset_size)
                
                epoch_policy_loss = 0
                epoch_value_loss = 0
                epoch_entropy = 0
                
                # 미니배치 반복
                for start_idx in range(0, dataset_size, minibatch_size):
                    # 미니배치 인덱스 선택
                    end_idx = min(start_idx + minibatch_size, dataset_size)
                    mb_indices = indices[start_idx:end_idx]
                    
                    # 미니배치 데이터 선택
                    mb_states = states[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    
                    # 현재 정책의 액션 확률과 가치 예측
                    action_probs, values = self.policy(mb_states)
                    values = values.squeeze()
                    
                    # 액션 분포 생성 및 액션 로그 확률 계산
                    dist = torch.distributions.Categorical(action_probs)
                    
                    # 다차원 액션 처리용 인덱스 생성
                    action_indices = torch.argmax(mb_actions, dim=1)
                    
                    # 현재 로그 확률 계산
                    new_log_probs = dist.log_prob(action_indices)
                    
                    # 엔트로피 계산 (정책의 다양성 측정)
                    entropy = dist.entropy().mean()
                    
                    # 정책 손실 계산 (PPO 목적 함수)
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    
                    # 비율 클리핑 (NaN/Inf 방지)
                    ratio = torch.clamp(ratio, 0.01, 10.0)
                    
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                    
                    # PPO 클리핑 손실 (최소값 선택)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 가치 함수 손실 (MSE와 smooth L1 loss 조합)
                    # MSE는 큰 오류에 민감, Smooth L1은 이상치에 덜 민감
                    value_mse = self.MseLoss(values, mb_returns)
                    value_smooth_l1 = F.smooth_l1_loss(values, mb_returns)
                    value_loss = 0.5 * value_mse + 0.5 * value_smooth_l1
                    
                    # 손실 클리핑 (학습 안정화)
                    if value_loss > max_value_loss:
                        value_loss = torch.clamp(value_loss, 0, max_value_loss)
                        
                    if policy_loss > max_policy_loss:
                        policy_loss = torch.clamp(policy_loss, 0, max_policy_loss)
                    
                    # 총 손실 = 정책 손실 + c1 * 가치 손실 - c2 * 엔트로피
                    loss = policy_loss + CRITIC_COEF * value_loss - ENTROPY_COEF * entropy
                    
                    # 그래디언트 계산 및 업데이트
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 그래디언트 클리핑 적용 (안정화)
                    if GRADIENT_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRADIENT_CLIP)
                    
                    self.optimizer.step()
                    
                    # 통계 기록
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy += entropy.item()
                
                # 에폭 평균 손실 계산
                avg_epoch_policy_loss = epoch_policy_loss / n_minibatches
                avg_epoch_value_loss = epoch_value_loss / n_minibatches
                avg_epoch_entropy = epoch_entropy / n_minibatches
                
                # 에폭별 손실 누적
                total_policy_loss += avg_epoch_policy_loss
                total_value_loss += avg_epoch_value_loss
                total_entropy += avg_epoch_entropy
                
                # KL 발산 조기 종료 (옵션) - 정책이 너무 크게 변경되는 것 방지
                # 현재 에폭에서 평균 정책 손실이 크게 상승하면 업데이트 중단
                if epoch > 0 and avg_epoch_policy_loss > 1.5 * (total_policy_loss / (epoch+1)):
                    self.logger.debug(f"KL 발산 탐지: 에폭 {epoch+1}/{self.k_epochs}에서 업데이트 조기 종료")
                    break

            # 이전 정책 업데이트 (정책 가중치 복사)
            self.policy_old.load_state_dict(self.policy.state_dict())
            
            # EMA 모델 업데이트 (있는 경우)
            if self.use_ema:
                self.update_ema_model()
            
            # 업데이트당 평균 손실 계산
            avg_policy_loss = total_policy_loss / (self.k_epochs * n_minibatches)
            avg_value_loss = total_value_loss / (self.k_epochs * n_minibatches)
            avg_entropy = total_entropy / (self.k_epochs * n_minibatches)
            
            self.logger.debug(
                f"PPO 업데이트 완료: 정책 손실={avg_policy_loss:.6f}, "
                f"가치 손실={avg_value_loss:.6f}, 엔트로피={avg_entropy:.6f}"
            )
            
            # 메모리 사용 완료 후 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return avg_policy_loss
        except Exception as e:
            self.logger.error(f"PPO 업데이트 중 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return 0 