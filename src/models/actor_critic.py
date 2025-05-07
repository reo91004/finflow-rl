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
        """
        Actor-Critic 모델 초기화.

        Args:
            n_assets (int): 자산(주식) 수.
            n_features (int): 각 자산당 피처 수.
            hidden_dim (int, optional): 은닉층 크기.
        """
        super(ActorCritic, self).__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # 온도 파라미터: 정책 결정성을 제어하는 하이퍼파라미터
        # 낮은 온도 = 확률 분포가 더 결정적(spiky), 높은 온도 = 확률 분포가 더 균등
        self.temperature = torch.tensor(SOFTMAX_TEMPERATURE_INITIAL, dtype=torch.float32, device=DEVICE, requires_grad=False)
        self.temp_min = SOFTMAX_TEMPERATURE_MIN
        self.temp_decay = SOFTMAX_TEMPERATURE_DECAY

        # 1. 특성 추출 모듈 (LSTM 기반)
        self.lstm = nn.LSTM(
            input_size=n_features,  # 입력 특성 수
            hidden_size=hidden_dim,  # 은닉층 크기
            num_layers=1,  # 단일 레이어 LSTM
            batch_first=False,  # (seq_len, batch, features) 형식 
        )
        
        # 2. 자기 주의 메커니즘 (에셋 간 관계 학습)
        self.self_attention = SelfAttention(hidden_dim)
        
        # 3. 액터 네트워크 (정책 생성)
        # 입력 -> 복층 MLP -> 출력 구조 (표현력 강화)
        self.actor_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # 최종 출력 레이어: 각 자산별 처리 후 나온 feature들을 flatten하여 n_assets+1 (현금포함)개의 logit 생성
        self.actor_head = nn.Linear(self.n_assets * (hidden_dim // 2), self.n_assets + 1)
        
        # 드롭아웃 추가 (과적합 방지)
        self.dropout = nn.Dropout(0.2)
        
        # 4. 크리틱 네트워크 (가치 추정)
        # 모든 자산 정보를 통합하여 포트폴리오 가치 평가
        self.critic_head = nn.Linear(n_assets * hidden_dim, 1)
        
        # 가중치 초기화
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
            self.temperature.mul_(self.temp_decay)
            self.temperature.clamp_(min=self.temp_min)

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
            states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)

        # Feature Extraction: LSTM 통과
        # LSTM에서는 (seq_len, batch, input_size) 형태 필요
        # LSTM_input: (n_assets, batch_size, n_features)
        states_permuted = states.permute(1, 0, 2).contiguous()
        
        # 초기 히든 스테이트와 셀 스테이트는 기본값으로 사용
        lstm_out, (h_n, c_n) = self.lstm(states_permuted)
        
        # LSTM 출력: (n_assets, batch_size, hidden_dim)
        # 마지막 출력값 추출: (batch_size, n_assets, hidden_dim)
        lstm_features = lstm_out.permute(1, 0, 2).contiguous()

        # 자기주의 메커니즘 적용 (에셋 간 관계 모델링)
        if hasattr(self, 'self_attention'):
            # 자기주의 메커니즘 적용 (차원 맞추기)
            # self_attention 입력: (batch_size, seq_len, hidden_dim) = (batch_size, n_assets, hidden_dim)
            attention_out, _ = self.self_attention(lstm_features)  # context만 사용
            features = attention_out
        else:
            features = lstm_features

        # 가치 네트워크 (Critic) - LSTM 유지하여 변경 없음
        # 전체 특성을 평탄화하여 가치 함수 추정
        # 모든 자산의 정보를 통합하여 포트폴리오 가치 평가
        value_features = features.reshape(batch_size, -1)  # (batch_size, n_assets * hidden_dim)
        
        # 가치 헤드 통과
        value = self.critic_head(value_features)  # (batch_size, 1)
        
        # 정책 네트워크 (Actor) 개선 - 복잡성 증가
        # 추가 레이어를 통한 정책 표현력 강화
        policy_features = features  # (batch_size, n_assets, hidden_dim)
        
        # 차원 축소 레이어 (정보 압축) - 각 자산별로 적용
        x = F.leaky_relu(self.actor_layer1(policy_features))
        # 드롭아웃 적용 (과적합 방지)
        x = self.dropout(x)
        # 두 번째 레이어 통과 - 각 자산별로 적용
        x = F.leaky_relu(self.actor_layer2(x))  # (batch_size, n_assets, hidden_dim // 2)
        
        # 모든 자산의 정제된 특성을 flatten하여 최종 로짓 계산
        x_flat = x.reshape(batch_size, -1)  # (batch_size, n_assets * (hidden_dim // 2))
        logits = self.actor_head(x_flat)  # (batch_size, n_assets + 1)
        
        # 온도 조절된 소프트맥스 활성화 함수로 확률 변환
        # (batch_size, n_assets+1) - 현금 포함
        action_probs = F.softmax(logits / self.temperature, dim=-1)
        
        # 발산 방지를 위한 안전장치
        if torch.isnan(action_probs).any():
            # NaN 발생 시 균등 분포로 대체
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        
        return action_probs, value.squeeze(-1)

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
                state_value.item()
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
                   - state_values (torch.Tensor): 상태 가치 [batch_size] 형태.
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