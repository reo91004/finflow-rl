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
    CRITIC_COEF,
    DEFAULT_ENTROPY_COEF
)
from src.models.running_mean_std import RunningMeanStd

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
        self.entropy_coef = DEFAULT_ENTROPY_COEF  # 엔트로피 계수 추가
        self.gradient_clip = GRADIENT_CLIP
        self.critic_coef = CRITIC_COEF
        self.batch_size = BATCH_SIZE
        self.lr = lr
        
        # 학습 안정화를 위한 설정
        self.gae_lambda = LAMBDA_GAE
        self.clip_grad_norm = True

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
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-5)
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
        
        # 모델 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """모델 가중치를 적절하게 초기화합니다."""
        for module in self.policy.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # He 초기화 (ReLU 활성화 함수에 적합)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
        raw_rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_raw_reward = 0
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
                episode_reward += reward
                episode_raw_reward += info.get("raw_reward", reward)

                if terminated or truncated:
                    done = True
                else:
                    state = next_state

            total_reward += episode_reward
            raw_rewards.append(episode_raw_reward)

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
        학습된 모델을 저장합니다.

        Args:
            episode: 현재 에피소드 번호
            reward: 검증 보상 값
            save_file: 저장할 파일 경로 (None이면 자동 생성)
        """
        if save_file is None:
            # 파일명 자동 생성
            if reward > self.best_reward:
                # 최고 성능 모델 저장
                save_file = os.path.join(self.model_path, "best_model.pth")
                self.best_reward = reward
                self.logger.info(f"새로운 최고 성능! 보상: {reward:.4f} -> {save_file}")
            else:
                # 체크포인트 저장
                checkpoint_dir = os.path.join(self.model_path, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_file = os.path.join(checkpoint_dir, f"model_ep{episode}.pth")

        # 저장할 모델과 데이터 준비
        if self.use_ema:
            # EMA 모델이 있으면 EMA 모델 저장
            model_state_dict = self.policy_ema.state_dict()
        else:
            # 없으면 현재 모델 저장
            model_state_dict = self.policy.state_dict()

        # 추가 정보와 함께 저장
        save_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "reward": reward,
            "episode": episode,
            "obs_rms": self.obs_rms if self.obs_rms is not None else None,
            "n_assets": self.n_assets,
            "n_features": self.n_features,
            "hidden_dim": self.policy.hidden_dim if hasattr(self.policy, "hidden_dim") else DEFAULT_HIDDEN_DIM,
        }

        try:
            # 모델 저장
            torch.save(save_dict, save_file)
            self.logger.info(f"모델 저장 완료: {save_file} (보상: {reward:.4f})")
            return True
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {e}")
            traceback.print_exc()
            return False

    def load_model(self, model_file=None):
        """
        저장된 모델을 로드합니다.

        Args:
            model_file: 로드할 모델 파일 경로

        Returns:
            bool: 로드 성공 여부
        """
        if model_file is None:
            # 모델 파일 자동 선택
            # 먼저 best_model.pth 찾기
            best_model_path = os.path.join(self.model_path, "best_model.pth")
            if os.path.exists(best_model_path):
                model_file = best_model_path
            else:
                # 없으면 가장 최근 체크포인트 찾기
                checkpoint_dir = os.path.join(self.model_path, "checkpoints")
                if os.path.exists(checkpoint_dir):
                    checkpoints = [
                        os.path.join(checkpoint_dir, f)
                        for f in os.listdir(checkpoint_dir)
                        if f.startswith("model_ep") and f.endswith(".pth")
                    ]
                    if checkpoints:
                        # 에피소드 번호 기준 정렬
                        checkpoints.sort(
                            key=lambda x: int(
                                os.path.basename(x)
                                .replace("model_ep", "")
                                .replace(".pth", "")
                            )
                        )
                        model_file = checkpoints[-1]  # 가장 최근 체크포인트

        if model_file is None or not os.path.exists(model_file):
            self.logger.error(f"모델 파일을 찾을 수 없음: {model_file}")
            return False

        try:
            # 모델 로드
            state_dict = torch.load(model_file, map_location=DEVICE)
            
            # 모델 구조 검증
            saved_n_assets = state_dict.get("n_assets", self.n_assets)
            saved_n_features = state_dict.get("n_features", self.n_features)
            
            if saved_n_assets != self.n_assets or saved_n_features != self.n_features:
                self.logger.warning(
                    f"모델 구조 불일치: 저장된 모델({saved_n_assets} 자산, {saved_n_features} 피처) vs "
                    f"현재 모델({self.n_assets} 자산, {self.n_features} 피처)"
                )
                # 구조 불일치 시 모델 재생성
                hidden_dim = state_dict.get("hidden_dim", DEFAULT_HIDDEN_DIM)
                self.policy = ActorCritic(self.n_assets, self.n_features, hidden_dim).to(DEVICE)
                self.policy_old = ActorCritic(self.n_assets, self.n_features, hidden_dim).to(DEVICE)
                
                if self.use_ema:
                    self.policy_ema = ActorCritic(self.n_assets, self.n_features, hidden_dim).to(DEVICE)
                    for param in self.policy_ema.parameters():
                        param.requires_grad = False
                
                self.logger.info(f"모델 구조 조정: {hidden_dim} 은닉층 차원으로 재생성됨")
                
                # 가중치 초기화만 진행하고 로딩은 무시
                self._init_weights()
                return True
            
            # 모델 가중치 로드
            self.policy.load_state_dict(state_dict["model_state_dict"])
            self.policy_old.load_state_dict(state_dict["model_state_dict"])
            
            if self.use_ema:
                self.policy_ema.load_state_dict(state_dict["model_state_dict"])
            
            # 옵티마이저 상태 로드 (있는 경우)
            if "optimizer_state_dict" in state_dict:
                try:
                    self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                except Exception as e:
                    self.logger.warning(f"옵티마이저 상태 로드 실패: {e}, 새로운 옵티마이저 사용")
            
            # RMS 통계 로드 (있는 경우)
            if "obs_rms" in state_dict and state_dict["obs_rms"] is not None:
                self.obs_rms = state_dict["obs_rms"]
            
            # 모델 로드 성공
            reward = state_dict.get("reward", "알 수 없음")
            episode = state_dict.get("episode", "알 수 없음")
            self.logger.info(f"모델 로드 성공: {model_file} (에피소드: {episode}, 보상: {reward})")
            
            return True
        
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류: {e}")
            traceback.print_exc()
            return False

    def select_action(self, state, use_ema=True):
        """
        주어진 상태에서 행동을 선택합니다.

        Args:
            state: 환경의 현재 상태
            use_ema: EMA 모델 사용 여부

        Returns:
            np.ndarray: 선택된 행동
        """
        if use_ema and self.use_ema:
            model = self.policy_ema
        else:
            model = self.policy_old

        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            state = state.unsqueeze(0)  # 배치 차원 추가
            action, _, _ = model.act(state)

        return action

    def compute_returns_and_advantages(self, memory):
        """
        GAE(Generalized Advantage Estimation)를 사용하여 returns와 advantages를 계산합니다.

        Args:
            memory: 학습 데이터가 담긴 메모리 객체

        Returns:
            tuple: (returns, advantages) 배열
        """
        # 메모리에서 데이터 추출
        rewards = memory.rewards
        is_terminals = memory.is_terminals
        values = memory.values
        
        returns = []
        advantages = []
        gae = 0
        
        # GAE 계산
        for step in reversed(range(len(rewards))):
            # 마지막 스텝이거나 에피소드가 종료된 경우, next_value는 0
            if step == len(rewards) - 1 or is_terminals[step]:
                next_value = 0
            else:
                next_value = values[step + 1]
            
            # TD 에러 계산
            delta = rewards[step] + self.gamma * next_value * (1 - is_terminals[step]) - values[step]
            
            # GAE(Generalized Advantage Estimation) 계산
            gae = delta + self.gamma * self.gae_lambda * (1 - is_terminals[step]) * gae
            
            # 장점과 반환값 계산
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        # 계산된 returns과 advantages를 텐서로 변환
        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        
        # Advantages 정규화 (학습 안정성 향상)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def update(self, memory):
        """
        메모리에 있는 데이터를 사용하여 정책을 업데이트합니다.

        Args:
            memory: 학습 데이터가 담긴 메모리 객체

        Returns:
            float: 평균 손실값
        """
        # 경험 부족 시 건너뛰기
        if len(memory.states) < self.batch_size:
            self.logger.warning(f"경험 부족으로 업데이트 건너뜀 (필요: {self.batch_size}, 현재: {len(memory.states)})")
            return 0.0

        # GAE를 사용하여 returns와 advantages 계산
        returns, advantages = self.compute_returns_and_advantages(memory)

        # 데이터 준비
        old_states = torch.FloatTensor(np.array(memory.states)).to(DEVICE)
        old_actions = torch.FloatTensor(np.array(memory.actions)).to(DEVICE)
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(DEVICE)
        
        # 배치 인덱스 생성
        batch_size = self.batch_size
        indices = np.arange(len(old_states))
        
        # 여러 에포크 동안 미니배치로 학습
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # 현재 학습률 확인
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            self.logger.warning(f"학습률이 매우 낮음: {current_lr:.8f}, 업데이트 건너뜀")
            return 0.0
        
        self.policy.train()
        
        for epoch in range(self.k_epochs):
            # 인덱스 셔플
            np.random.shuffle(indices)
            
            # 미니배치 반복
            for start_idx in range(0, len(indices), batch_size):
                # 배치 인덱스 선택
                idx = indices[start_idx:start_idx + batch_size]
                
                # 배치 데이터 추출
                batch_states = old_states[idx]
                batch_actions = old_actions[idx]
                batch_logprobs = old_logprobs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # 현재 정책의 로그 확률 및 엔트로피, 상태 가치 계산
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # 비율 계산 (importance sampling)
                ratios = torch.exp(logprobs - batch_logprobs.detach())
                
                # PPO 정책 손실 계산 (클리핑 적용)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 가치 함수 손실 계산
                value_loss = self.critic_coef * self.MseLoss(state_values, batch_returns)
                
                # 엔트로피 보너스 (탐색 촉진)
                entropy_loss = -self.entropy_coef * dist_entropy.mean()
                
                # 전체 손실 계산
                loss = policy_loss + value_loss + entropy_loss
                
                # 그래디언트 계산 및 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑 (선택적)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
                
                self.optimizer.step()
                
                # 손실 누적
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += dist_entropy.mean().item()
                total_loss += loss.item()
                n_updates += 1
        
        # 구 정책을 새 정책으로 업데이트
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # EMA 모델 업데이트
        if self.use_ema:
            self.update_ema_model()
        
        # 평균 손실 계산
        avg_loss = total_loss / max(1, n_updates)
        avg_policy_loss = total_policy_loss / max(1, n_updates)
        avg_value_loss = total_value_loss / max(1, n_updates)
        avg_entropy = total_entropy / max(1, n_updates)
        
        # 손실 로깅 (낮은 레벨로)
        self.logger.debug(
            f"PPO 업데이트 완료: 손실={avg_loss:.4f}, 정책손실={avg_policy_loss:.4f}, "
            f"가치손실={avg_value_loss:.4f}, 엔트로피={avg_entropy:.4f}, 배치={n_updates}개"
        )
        
        return avg_loss 