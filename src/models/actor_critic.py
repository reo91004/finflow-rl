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
import logging
from src.constants import (
    DEVICE, 
    DEFAULT_HIDDEN_DIM, 
    SOFTMAX_TEMPERATURE_INITIAL,
    SOFTMAX_TEMPERATURE_MIN,
    SOFTMAX_TEMPERATURE_DECAY
)

# 로거 설정
# logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """
    PPO를 위한 액터-크리틱(Actor-Critic) 네트워크입니다. (past.py 스타일)

    - 입력: 평탄화된 상태 (batch_size, n_assets * n_features)
    - LSTM: 시계열 패턴 포착을 위한 순환 레이어
    - 액터 출력: Softmax 기반 포트폴리오 분배 (온도 스케일링 적용)
    - 크리틱 출력: 상태 가치 (State Value)
    """

    def __init__(self, n_assets, n_features, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(ActorCritic, self).__init__()
        # self.input_dim = n_assets * n_features # past.py에서는 사용되지 않음
        self.n_assets_plus_cash = n_assets + 1  # 현금 자산 추가 (past.py의 self.n_assets에 해당)
        # self.n_features = n_features # past.py에서는 사용되지 않음
        self.hidden_dim = hidden_dim
        self.n_original_assets = n_assets # LSTM 처리 시 사용 (현금 제외 자산 수)

        # 온도 파라미터 (학습 가능) - past.py와 동일
        self.temperature = nn.Parameter(torch.tensor(SOFTMAX_TEMPERATURE_INITIAL))
        self.temp_min = SOFTMAX_TEMPERATURE_MIN
        self.temp_decay = SOFTMAX_TEMPERATURE_DECAY

        # LSTM 레이어 (시계열 패턴 포착) - past.py와 동일
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, batch_first=True
        ).to(DEVICE)

        # 공통 특징 추출 레이어 (past.py의 actor_critic_base와 동일하게 구성)
        self.actor_critic_base = nn.Sequential(
            nn.Linear(hidden_dim * n_assets, hidden_dim), # 입력은 n_original_assets * hidden_dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        ).to(DEVICE)

        # 액터 헤드 (로짓 출력) - past.py와 동일
        self.actor_head = nn.Linear(hidden_dim // 2, self.n_assets_plus_cash).to(DEVICE)

        # 크리틱 헤드 (상태 가치) - past.py와 동일
        self.critic_head = nn.Linear(hidden_dim // 2, 1).to(DEVICE)

        # 가중치 초기화 적용 - past.py와 동일
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ 신경망 가중치를 초기화합니다 (Kaiming He 초기화 사용). """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(
                module.weight, a=0, mode="fan_in", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        # LayerNorm 초기화는 기본값 사용 또는 필요시 추가

    def update_temperature(self):
        """학습 과정에서 온도 값을 점진적으로 감소시킵니다."""
        with torch.no_grad():
            self.temperature.mul_(self.temp_decay).clamp_(min=self.temp_min)

    def forward(self, states):
        """
        네트워크의 순전파를 수행합니다.

        Args:
            states (torch.Tensor): 입력 상태 텐서.
                                   (batch_size, n_assets, n_features) 또는 (n_assets, n_features) 형태.
                                   여기서 n_assets는 현금을 제외한 원본 자산 수.

        Returns:
            tuple: (action_probs, value)
                   - action_probs (torch.Tensor): 각 자산에 대한 투자 비중 확률 (현금 포함).
                   - value (torch.Tensor): 크리틱 헤드의 출력 (상태 가치).
        """
        batch_size = states.size(0) if states.dim() == 3 else 1

        if states.dim() == 2: # (n_assets, n_features)
            states = states.unsqueeze(0) # (1, n_assets, n_features)

        if torch.isnan(states).any() or torch.isinf(states).any():
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # LSTM 처리 (past.py 스타일)
        # 각 자산별로 피처 시퀀스를 LSTM에 통과시킴
        lstm_outputs = []
        # states: (batch_size, self.n_original_assets, n_features)
        for i in range(self.n_original_assets): # 현금 제외한 자산 수만큼 반복
            asset_feats = states[:, i, :].view(batch_size, 1, -1)  # (batch, 1, n_features)
            # LSTM은 batch_first=True이므로 (batch, seq_len, input_size)
            # 현재 asset_feats는 (batch, 1, n_features) -> 1은 seq_len 역할
            # 올바른 LSTM 입력 형태: (batch_size, seq_len, feature_size)
            # view 대신 squeeze/unsqueeze 사용이 더 명확할 수 있음: states[:, i, :].unsqueeze(1)
            lstm_out, _ = self.lstm(asset_feats)  # lstm_out: (batch, 1, hidden_dim)
            lstm_outputs.append(lstm_out[:, -1, :])  # 마지막 hidden state: (batch, hidden_dim)

        # 모든 자산의 LSTM 출력을 연결
        lstm_concat = torch.cat(lstm_outputs, dim=1)  # (batch, n_original_assets * hidden_dim)
        # lstm_flat = lstm_concat.reshape(batch_size, -1) # past.py에서는 이 변수명 사용. 여기서는 lstm_concat 직접 사용

        # 공통 베이스 네트워크 통과
        base_output = self.actor_critic_base(lstm_concat) # 입력 차원: n_original_assets * hidden_dim

        # 액터 출력: 로짓 계산
        logits = self.actor_head(base_output) # 입력 차원: hidden_dim // 2

        scaled_logits = logits / (self.temperature + 1e-8)
        action_probs = F.softmax(scaled_logits, dim=-1) # 출력 차원: n_assets_plus_cash
        action_probs = torch.clamp(action_probs, min=1e-7, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        # 크리틱 출력: 상태 가치
        value = self.critic_head(base_output) # 입력 차원: hidden_dim // 2

        return action_probs, value.squeeze(-1) # value를 (batch_size,) 형태로

    def act(self, state):
        """
        주어진 상태에 대해 행동(action), 로그 확률(log_prob), 상태 가치(value)를 반환합니다.
        추론(inference) 시 사용됩니다. (past.py와 거의 동일)
        """
        if isinstance(state, np.ndarray):
            if state.ndim == 2: # (n_assets, n_features)
                 state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            # elif state.ndim == 1: # past.py에는 있었으나, 여기서는 상태가 항상 (n_assets, n_features)라고 가정
            #      state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            else:
                 raise ValueError(f"act 메서드: 예상치 못한 NumPy 상태 차원: {state.shape}")
        elif torch.is_tensor(state):
             if state.dim() == 2: # (n_assets, n_features)
                  state_tensor = state.float().unsqueeze(0).to(DEVICE)
             # elif state.dim() == 1:
             #      state_tensor = state.float().unsqueeze(0).to(DEVICE)
             else:
                  raise ValueError(f"act 메서드: 예상치 못한 Tensor 상태 차원: {state.shape}")
        else:
            raise TypeError(f"act 메서드: 지원하지 않는 상태 타입: {type(state)}")

        self.eval() # 평가 모드
        with torch.no_grad():
            action_probs, value = self.forward(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            action = torch.zeros_like(action_probs)
            action.scatter_(1, action_idx.unsqueeze(-1), 1.0)
        self.train() # 다시 학습 모드
        
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def evaluate(self, states, actions):
        """
        주어진 상태(states)와 행동(actions)에 대한 로그 확률(log_prob),
        분포 엔트로피(entropy), 상태 가치(value)를 계산합니다.
        PPO 업데이트 시 사용됩니다. (past.py와 거의 동일)
        """
        action_probs, value = self.forward(states)
        
        if actions.size(-1) == action_probs.size(-1):
            actions_idx = torch.argmax(actions, dim=-1)
        else: # 이미 인덱스 형태일 경우
            actions_idx = actions
        
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(actions_idx)
        entropy = dist.entropy()

        return log_prob, entropy, value # value는 이미 forward에서 squeeze(-1) 되었음 