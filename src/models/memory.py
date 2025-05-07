"""
경험 리플레이 메모리 모듈

PPO 알고리즘의 학습에 필요한 경험(transition)을 저장하고 관리하는 메모리 클래스를 구현합니다.
상태, 행동, 로그 확률, 보상, 종료 여부, 상태 가치 등의 정보를 저장합니다.
"""

import torch
import numpy as np
from src.constants import DEVICE

class Memory:
    """
    PPO 학습을 위한 경험(Experience) 저장 버퍼입니다.
    NumPy 기반으로 상태, 행동, 로그 확률, 보상, 종료 여부, 상태 가치를 저장합니다.
    추가로 원시 보상(정규화 전 보상)도 저장하여 학습에 활용할 수 있습니다.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.raw_rewards = []  # 정규화 전 원시 보상 저장용
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        """메모리에 저장된 모든 경험을 삭제합니다."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.raw_rewards[:]  # 원시 보상도 초기화
        del self.is_terminals[:]
        del self.values[:]

    def add_experience(self, state, action, logprob, reward, is_terminal, value, raw_reward=None):
        """
        새로운 경험을 메모리에 추가합니다.
        
        Args:
            state: 상태
            action: 행동
            logprob: 행동의 로그 확률
            reward: 정규화된 보상
            is_terminal: 종료 여부
            value: 상태 가치
            raw_reward: 정규화 전 원시 보상 (None이면 reward와 동일하게 처리)
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        # raw_reward가 None이면 reward와 동일하게 사용
        self.raw_rewards.append(raw_reward if raw_reward is not None else reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value) 
        
    def __len__(self):
        """메모리에 저장된 경험의 개수를 반환합니다."""
        return len(self.states)
        
    def get_states_tensor(self):
        """상태 배열을 텐서로 변환하여 반환합니다."""
        return torch.tensor(np.array(self.states), dtype=torch.float32)
        
    def get_actions_tensor(self):
        """행동 배열을 텐서로 변환하여 반환합니다."""
        return torch.tensor(np.array(self.actions), dtype=torch.float32)
        
    def get_log_probs_tensor(self):
        """로그 확률 배열을 텐서로 변환하여 반환합니다."""
        return torch.tensor(np.array(self.logprobs), dtype=torch.float32)
        
    def get_rewards_tensor(self):
        """보상 배열을 텐서로 변환하여 반환합니다."""
        return torch.tensor(np.array(self.rewards), dtype=torch.float32)
        
    def get_raw_rewards_tensor(self):
        """원시 보상 배열을 텐서로 변환하여 반환합니다."""
        return torch.tensor(np.array(self.raw_rewards), dtype=torch.float32)
        
    def get_is_terminals_tensor(self):
        """종료 여부 배열을 텐서로 변환하여 반환합니다."""
        return torch.tensor(np.array(self.is_terminals), dtype=torch.float32)
        
    def get_values_tensor(self):
        """상태 가치 배열을 텐서로 변환하여 반환합니다."""
        return torch.tensor(np.array(self.values), dtype=torch.float32) 