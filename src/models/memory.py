"""
강화학습 경험 저장 메모리 모듈

PPO 학습에 사용되는 경험 데이터를 저장하고 관리하는 기능을 제공합니다.
"""

import numpy as np
import torch

class Memory:
    """
    PPO 에이전트의 경험을 저장하는 메모리 클래스입니다.
    상태, 행동, 로그 확률, 보상, 종료 여부 등을 저장합니다.
    학습 시 BATCH_SIZE에 맞게 데이터를 샘플링하는 기능도 제공합니다.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.raw_rewards = []  # 정규화되지 않은 원시 보상 저장
        
    def clear_memory(self):
        """메모리의 모든 내용을 지웁니다."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
        del self.raw_rewards[:]
        
    def add_experience(self, state, action, logprob, reward, is_terminal, value, raw_reward=None):
        """
        새로운 경험을 메모리에 추가합니다.
        
        Args:
            state: 환경의 현재 상태
            action: 에이전트가 선택한 행동
            logprob: 행동의 로그 확률
            reward: 환경에서 받은 정규화된 보상
            is_terminal: 에피소드 종료 여부
            value: 크리틱이 예측한 상태 가치
            raw_reward: 정규화되지 않은 원시 보상 (기본값: None)
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value)
        
        # raw_reward가 제공되지 않은 경우 정규화된 reward 사용
        self.raw_rewards.append(raw_reward if raw_reward is not None else reward)
        
    def get_batch(self, batch_size, device):
        """
        메모리에서 미니배치를 샘플링합니다.
        
        Args:
            batch_size: 배치 크기
            device: 텐서를 로드할 장치 (CPU 또는 GPU)
            
        Returns:
            tuple: (states, actions, logprobs, rewards, is_terminals, values, raw_rewards)
                   배치 크기만큼 무작위로 샘플링된 경험 데이터
        """
        indices = np.random.randint(0, len(self.states), batch_size)
        
        states_batch = [self.states[i] for i in indices]
        actions_batch = [self.actions[i] for i in indices]
        logprobs_batch = [self.logprobs[i] for i in indices]
        rewards_batch = [self.rewards[i] for i in indices]
        is_terminals_batch = [self.is_terminals[i] for i in indices]
        values_batch = [self.values[i] for i in indices]
        raw_rewards_batch = [self.raw_rewards[i] for i in indices]
        
        # 텐서로 변환
        states_tensor = torch.FloatTensor(np.array(states_batch)).to(device)
        actions_tensor = torch.FloatTensor(np.array(actions_batch)).to(device)
        logprobs_tensor = torch.FloatTensor(np.array(logprobs_batch)).to(device)
        rewards_tensor = torch.FloatTensor(np.array(rewards_batch)).to(device)
        is_terminals_tensor = torch.FloatTensor(np.array(is_terminals_batch)).to(device)
        values_tensor = torch.FloatTensor(np.array(values_batch)).to(device)
        raw_rewards_tensor = torch.FloatTensor(np.array(raw_rewards_batch)).to(device)
        
        return (
            states_tensor,
            actions_tensor,
            logprobs_tensor,
            rewards_tensor,
            is_terminals_tensor,
            values_tensor,
            raw_rewards_tensor
        )
    
    def get_all_as_tensors(self, device):
        """
        모든 경험 데이터를 텐서로 변환하여 반환합니다.
        
        Args:
            device: 텐서를 로드할 장치 (CPU 또는 GPU)
            
        Returns:
            tuple: (states, actions, logprobs, rewards, is_terminals, values, raw_rewards)
                   텐서로 변환된 모든 경험 데이터
        """
        states_tensor = torch.FloatTensor(np.array(self.states)).to(device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(device)
        logprobs_tensor = torch.FloatTensor(np.array(self.logprobs)).to(device)
        rewards_tensor = torch.FloatTensor(np.array(self.rewards)).to(device)
        is_terminals_tensor = torch.FloatTensor(np.array(self.is_terminals)).to(device)
        values_tensor = torch.FloatTensor(np.array(self.values)).to(device)
        raw_rewards_tensor = torch.FloatTensor(np.array(self.raw_rewards)).to(device)
        
        return (
            states_tensor,
            actions_tensor,
            logprobs_tensor,
            rewards_tensor,
            is_terminals_tensor,
            values_tensor,
            raw_rewards_tensor
        )
    
    def get_raw_rewards_tensor(self, device=None):
        """
        정규화되지 않은 원시 보상을 텐서로 변환하여 반환합니다.
        
        Args:
            device: 텐서를 로드할 장치 (기본값: None)
            
        Returns:
            torch.Tensor: 원시 보상 텐서
        """
        if device is None:
            return torch.FloatTensor(self.raw_rewards)
        return torch.FloatTensor(self.raw_rewards).to(device)
    
    def __len__(self):
        """메모리에 저장된 경험의 수를 반환합니다."""
        return len(self.states) 