"""
Actor-Critic 모델 모듈

PPO 알고리즘에서 사용되는 Actor-Critic 네트워크 아키텍처를 구현합니다.
액터(정책)와 크리틱(가치 함수)이 특성 추출기를 공유하는 구조입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import (
    DEVICE,
    SOFTMAX_TEMPERATURE_INITIAL,
    SOFTMAX_TEMPERATURE_MIN,
    SOFTMAX_TEMPERATURE_DECAY
)

class ActorCritic(nn.Module):
    """
    Actor-Critic 네트워크 클래스입니다.
    
    특성:
    - 공유 특성 추출기 사용 (특성 공유를 통한 학습 효율성)
    - 액터(Actor)는 행동 확률 분포 출력
    - 크리틱(Critic)은 상태 가치 예측
    - 파라미터 초기화 및 온도 스케일링 적용
    """

    def __init__(self, n_assets, n_features, hidden_dim=128):
        """
        Args:
            n_assets: 자산의 수
            n_features: 각 자산의 특성 수
            hidden_dim: 은닉층의 차원
        """
        super(ActorCritic, self).__init__()
        
        self.n_assets = n_assets
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.action_dim = n_assets + 1  # 각 자산 + 현금
        
        # 온도 스케일링 파라미터 (소프트맥스의 온도 조절)
        self.temperature = SOFTMAX_TEMPERATURE_INITIAL
        self.min_temperature = SOFTMAX_TEMPERATURE_MIN
        self.temperature_decay = SOFTMAX_TEMPERATURE_DECAY
        
        # 특성 추출기 (Feature Extractor)
        # 각 자산의 특성을 처리하기 위한 개별 네트워크
        self.asset_feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.LayerNorm(hidden_dim // 2),
            ) for _ in range(n_assets)
        ])
        
        # 공유 특성 처리기 (Shared Feature Processor)
        # 모든 자산의 처리된 특성을 합쳐서 처리
        self.shared_feature_processor = nn.Sequential(
            nn.Linear(n_assets * hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 액터 네트워크 (포트폴리오 가중치 예측)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.action_dim)
        )
        
        # 크리틱 네트워크 (가치 함수 예측)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # 모델 가중치 초기화
        self._init_weights()
    
    def _init_weights(self, module=None):
        """
        네트워크 가중치를 초기화합니다.
        
        Args:
            module: 초기화할 모듈 (None이면 전체 모델)
        """
        if module is None:
            modules = self.modules()
        else:
            modules = [module]
            
        for module in modules:
            if isinstance(module, nn.Linear):
                # He 초기화 (ReLU 계열 활성화 함수에 적합)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def update_temperature(self):
        """
        소프트맥스 온도 파라미터를 업데이트합니다.
        (학습이 진행됨에 따라 온도를 낮추어 더 확정적인 행동을 선택하도록 함)
        """
        self.temperature = max(self.min_temperature, 
                              self.temperature * self.temperature_decay)
        
    def _extract_features(self, state):
        """
        상태에서 특성을 추출합니다.
        
        Args:
            state: 모델 입력 상태 (배치 크기 x n_assets x n_features)
            
        Returns:
            torch.Tensor: 추출된 특성 (배치 크기 x hidden_dim)
        """
        # 입력 차원 검사 및 조정
        batch_size = state.size(0)
        
        # 각 자산별로 특성 추출
        asset_features = []
        for i in range(self.n_assets):
            asset_feature = self.asset_feature_extractors[i](state[:, i])
            asset_features.append(asset_feature)
        
        # 모든 자산 특성 결합
        combined_features = torch.cat(asset_features, dim=1)
        
        # 공유 특성 처리
        shared_features = self.shared_feature_processor(combined_features)
        
        return shared_features
    
    def forward(self, state):
        """
        주어진 상태에 대해 액터와 크리틱 값을 계산합니다.
        
        Args:
            state: 모델 입력 상태 (배치 크기 x n_assets x n_features)
            
        Returns:
            tuple: (action_probs, state_value)
                  - action_probs: 행동 확률 분포
                  - state_value: 상태 가치 예측
        """
        # 특성 추출
        features = self._extract_features(state)
        
        # 액터: 행동 확률 계산
        action_logits = self.actor(features)
        
        # 온도 스케일링 적용한 소프트맥스
        action_probs = F.softmax(action_logits / self.temperature, dim=-1)
        
        # 크리틱: 상태 가치 계산
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def act(self, state):
        """
        상태에 기반하여 행동을 선택합니다.
        
        Args:
            state: 환경의 현재 상태
            
        Returns:
            tuple: (action, log_prob, value)
                  - action: 선택된 행동
                  - log_prob: 행동의 로그 확률
                  - value: 상태 가치 예측
        """
        # 그래디언트 계산 없이 추론만 수행
        with torch.no_grad():
            # 상태를 텐서로 변환
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(DEVICE)
                
            # 배치 차원 확인 및 추가
            if state.dim() == 2:
                state = state.unsqueeze(0)
                
            # 액터-크리틱 네트워크 통과
            action_probs, state_value = self.forward(state)
            
            # 확률적 행동 선택 (카테고리컬 분포에서 샘플링)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            
            # 원-핫 인코딩으로 변환 (포트폴리오 가중치 형태)
            action = torch.zeros_like(action_probs)
            action.scatter_(1, action_idx.unsqueeze(-1), 1.0)
            
            # 선택된 행동의 로그 확률
            log_prob = dist.log_prob(action_idx)
            
        # NumPy 배열로 변환하여 반환
        action_np = action.squeeze(0).cpu().numpy()
        log_prob_np = log_prob.squeeze(0).cpu().numpy()
        value_np = state_value.squeeze(0).cpu().numpy()
        
        return action_np, log_prob_np, value_np
    
    def evaluate(self, states, actions):
        """
        주어진 상태와 행동에 대한 로그 확률, 상태 가치, 엔트로피를 계산합니다.
        
        Args:
            states: 상태 배치
            actions: 행동 배치
            
        Returns:
            tuple: (logprobs, state_values, dist_entropy)
                  - logprobs: 행동의 로그 확률
                  - state_values: 상태 가치 예측
                  - dist_entropy: 행동 분포의 엔트로피
        """
        # 액터-크리틱 네트워크 통과
        action_probs, state_values = self.forward(states)
        
        # 행동 분포 생성
        dist = torch.distributions.Categorical(action_probs)
        
        # 행동 인덱스 추출 (원-핫 인코딩된 행동에서)
        action_indices = torch.argmax(actions, dim=1)
        
        # 로그 확률 계산
        logprobs = dist.log_prob(action_indices)
        
        # 엔트로피 계산 (탐색 정도를 측정)
        dist_entropy = dist.entropy()
        
        return logprobs, state_values.squeeze(-1), dist_entropy 