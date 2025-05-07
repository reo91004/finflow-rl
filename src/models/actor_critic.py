"""
Actor-Critic 신경망 모델 모듈

PPO 알고리즘을 위한 Actor-Critic 네트워크를 구현합니다.
LSTM을 활용한 시계열 처리와 Softmax 온도 스케일링을 통한 확률 분포 조정 기능을 포함합니다.
상태를 입력으로 받아 행동 확률 분포와 상태 가치를 출력합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import (
    DEVICE, 
    DEFAULT_HIDDEN_DIM, 
    SOFTMAX_TEMPERATURE_INITIAL,
    SOFTMAX_TEMPERATURE_MIN,
    SOFTMAX_TEMPERATURE_DECAY
)

class SelfAttention(nn.Module):
    """
    시계열 데이터를 위한 자기 주의(Self-Attention) 메커니즘입니다.
    LSTM 출력에 적용하여 중요한 패턴에 가중치를 부여합니다.
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        q = self.query(x)  # (batch, seq_len, hidden_dim)
        k = self.key(x)    # (batch, seq_len, hidden_dim)
        v = self.value(x)  # (batch, seq_len, hidden_dim)
        
        # 어텐션 점수 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)
        
        # 소프트맥스로 어텐션 가중치 계산
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # 가중합 계산
        context = torch.matmul(attention_weights, v)  # (batch, seq_len, hidden_dim)
        
        return context, attention_weights

class ActorCritic(nn.Module):
    """
    PPO를 위한 액터-크리틱(Actor-Critic) 네트워크입니다.

    - 입력: 평탄화된 상태 (batch_size, n_assets * n_features)
    - LSTM: 시계열 패턴 포착을 위한 다층 순환 레이어
    - 어텐션: 중요한 패턴에 집중하는 자기 주의 메커니즘 추가
    - 액터 출력: Softmax 기반 포트폴리오 분배 (온도 스케일링 적용)
    - 크리틱 출력: 상태 가치 (State Value)
    """

    def __init__(self, n_assets, n_features, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(ActorCritic, self).__init__()
        self.input_dim = n_assets * n_features
        self.n_assets = n_assets + 1  # 현금 자산 추가
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # 온도 파라미터 (학습 가능)
        self.temperature = nn.Parameter(torch.tensor(SOFTMAX_TEMPERATURE_INITIAL))
        self.temp_min = SOFTMAX_TEMPERATURE_MIN
        self.temp_decay = SOFTMAX_TEMPERATURE_DECAY

        # 다층 LSTM 레이어 (시계열 패턴 포착 강화)
        self.lstm_layers = 2  # LSTM 레이어 수 증가 (1 -> 2)
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=0.2,  # 드롭아웃 추가로 과적합 방지
            bidirectional=True  # 양방향 LSTM으로 성능 향상
        ).to(DEVICE)
        
        # 양방향 LSTM이므로 출력 차원이 2배
        self.lstm_output_dim = hidden_dim * 2
        
        # 자기 주의(Self-Attention) 메커니즘 추가
        self.attention = SelfAttention(self.lstm_output_dim).to(DEVICE)
        
        # 자산별 특징 압축 레이어
        self.asset_compression = nn.Sequential(
            nn.Linear(self.lstm_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(DEVICE)

        # 공통 특징 추출 레이어 (더 깊고 넓게)
        self.actor_critic_base = nn.Sequential(
            nn.Linear(hidden_dim * n_assets, hidden_dim * 2),  # 더 넓은 레이어
            nn.BatchNorm1d(hidden_dim * 2),  # 배치 정규화로 변경
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 배치 정규화로 변경
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # 배치 정규화로 변경
            nn.ReLU(),
        ).to(DEVICE)

        # 액터 헤드 (로짓 출력)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_assets)
        ).to(DEVICE)

        # 크리틱 헤드 (상태 가치)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        ).to(DEVICE)

        # 가중치 초기화 적용
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """신경망 가중치를 초기화합니다 (Kaiming He 초기화 사용)."""
        if isinstance(module, nn.Linear):
            # ReLU 활성화 함수에 적합한 Kaiming 초기화
            nn.init.kaiming_uniform_(
                module.weight, a=0, mode="fan_in", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)  # 편향은 0으로 초기화
        elif isinstance(module, nn.LSTM):
            # LSTM 초기화
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            # 배치 정규화 초기화
            nn.init.constant_(module.weight, 1.0)  # 가중치를 1로 초기화
            nn.init.constant_(module.bias, 0.0)  # 편향을 0으로 초기화

    def update_temperature(self):
        """학습 과정에서 온도 값을 점진적으로 감소시킵니다."""
        with torch.no_grad():
            # 최소값보다 작아지지 않도록 조정
            self.temperature.mul_(self.temp_decay).clamp_(min=self.temp_min)

    def forward(self, states):
        """
        네트워크의 순전파를 수행합니다.

        Args:
            states (torch.Tensor): 입력 상태 텐서.
                                   (batch_size, n_assets, n_features) 또는 (n_assets, n_features) 형태.

        Returns:
            tuple: (action_probs, value)
                   - action_probs (torch.Tensor): 각 자산에 대한 투자 비중 확률.
                   - value (torch.Tensor): 크리틱 헤드의 출력 (상태 가치).
        """
        batch_size = states.size(0) if states.dim() == 3 else 1

        # 단일 상태인 경우 배치 차원 추가
        if states.dim() == 2:
            states = states.unsqueeze(0)

        # NaN/Inf 입력 방지 (안정성 강화)
        if torch.isnan(states).any() or torch.isinf(states).any():
            # logger.warning(f"ActorCritic 입력에 NaN/Inf 발견. 0으로 대체합니다. Shape: {states.shape}")
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # LSTM 처리 및 어텐션 적용
        lstm_outputs = []
        
        # 각 자산별로 피처 시퀀스를 LSTM에 통과시킴
        for i in range(states.size(1)):
            # (batch_size, 1, n_features) 형태로 재구성
            asset_feats = states[:, i, :].view(batch_size, 1, -1)
            
            # LSTM으로 처리
            # (batch, 1, lstm_output_dim)
            lstm_out, _ = self.lstm(asset_feats)
            
            # 어텐션 메커니즘 적용
            # (batch, 1, lstm_output_dim)
            context, _ = self.attention(lstm_out)
            
            # 마지막 시퀀스 출력 추출 (batch, lstm_output_dim)
            asset_out = context[:, -1, :]
            
            # 자산별 특징 압축 (배치 정규화를 위해 차원 변경)
            asset_out_flattened = asset_out.reshape(batch_size, -1)
            compressed = self.asset_compression(asset_out_flattened)
            
            lstm_outputs.append(compressed)

        # 모든 자산의 특징을 연결
        lstm_concat = torch.cat(lstm_outputs, dim=1)  # (batch, n_assets*hidden_dim)
        lstm_flat = lstm_concat.reshape(batch_size, -1)  # 평탄화

        # 공통 베이스 네트워크 통과
        base_output = self.actor_critic_base(lstm_flat)

        # 액터 출력: 로짓 계산
        logits = self.actor_head(base_output)

        # 온도 스케일링 적용 (낮은 온도 = 더 높은 분산)
        # 온도가 낮을수록 확률 분포는 더 극단적으로 변환됨 (Sparsity 유도)
        scaled_logits = logits / (self.temperature + 1e-8)

        # Softmax로 확률 분포 계산
        action_probs = F.softmax(scaled_logits, dim=-1)

        # 수치적 안정성을 위한 클리핑
        action_probs = torch.clamp(action_probs, min=1e-7, max=1.0)

        # 확률 합이 1이 되도록 정규화
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        # 크리틱 출력: 상태 가치
        value = self.critic_head(base_output)

        return action_probs, value

    def act(self, state):
        """
        주어진 상태에 대해 행동(action), 로그 확률(log_prob), 상태 가치(value)를 반환합니다.
        추론(inference) 시 사용됩니다.

        Args:
            state (np.ndarray): 현재 환경 상태 (정규화된 값).

        Returns:
            tuple: (action, log_prob, value)
                   - action (np.ndarray): 샘플링된 행동 (자산 비중).
                   - log_prob (float): 샘플링된 행동의 로그 확률.
                   - value (float): 예측된 상태 가치.
        """
        # NumPy 배열을 Tensor로 변환하고 배치 차원 추가
        if isinstance(state, np.ndarray):
            # 올바른 형태로 변환 (n_assets, n_features) -> (1, n_assets, n_features) 가정?
            if state.ndim == 2:  # (n_assets, n_features)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            elif state.ndim == 1:  # 이미 평탄화된 경우? (호환성 위해)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            else:
                raise ValueError(f"Unsupported state shape: {state.shape}")
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 2 else state

        # 평가 모드에서 실행 (배치 정규화 등을 위해)
        self.eval()
        
        with torch.no_grad():
            # 네트워크로 액션 확률과 가치 계산
            action_probs, state_value = self.forward(state_tensor)

            # 액션 확률 분포 생성
            dist = torch.distributions.Categorical(action_probs)
            
            # 액션 샘플링
            action_idx = dist.sample()
            
            # 원-핫 인코딩으로 변환
            action = torch.zeros_like(action_probs)
            action.scatter_(1, action_idx.unsqueeze(-1), 1.0)
            
            # 로그 확률 계산
            log_prob = dist.log_prob(action_idx)
            
            # 학습 모드로 복원
            self.train()
            
            # NumPy로 변환하여 반환
            return (
                action.squeeze(0).cpu().numpy(),
                log_prob.item(),
                state_value.squeeze().item()
            )

    def evaluate(self, states, actions):
        """
        주어진 상태와 액션에 대한 로그 확률, 엔트로피, 상태 가치를 계산합니다.
        PPO 업데이트 시 사용됩니다.

        Args:
            states (torch.Tensor): 상태 배치 텐서.
            actions (torch.Tensor): 액션 배치 텐서.

        Returns:
            tuple: (log_probs, entropy, state_values)
                   - log_probs (torch.Tensor): 액션의 로그 확률.
                   - entropy (torch.Tensor): 정책의 엔트로피.
                   - state_values (torch.Tensor): 상태 가치.
        """
        # 정책과 가치 함수 출력
        action_probs, state_values = self.forward(states)
        
        # 디스트리뷰션 생성
        dist = torch.distributions.Categorical(action_probs)
        
        # 액션 인덱스 추출 (원-핫 인코딩된 액션에서)
        action_indices = torch.argmax(actions, dim=1)
        
        # 로그 확률 계산
        log_probs = dist.log_prob(action_indices)
        
        # 엔트로피 계산
        entropy = dist.entropy()
        
        return log_probs, entropy, state_values 