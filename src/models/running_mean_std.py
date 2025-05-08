"""
이동 평균과 표준편차 계산 모듈

Welford의 알고리즘을 사용하여 상태 및 보상의 정규화를 위한 이동 평균과 표준편차를 계산하는 클래스를 구현합니다.
주로 강화학습 환경에서 입력 상태나 보상의 스케일을 정규화하기 위해 사용됩니다.
"""

import numpy as np
from src.constants import RMS_EPSILON

class RunningMeanStd:
    """
    Welford's online algorithm을 사용하여 이동 평균과 표준편차를 계산합니다.
    상태 및 보상 정규화에 사용됩니다.
    Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, epsilon=RMS_EPSILON, shape=()):
        """
        이동 평균 및 표준편차 계산기를 초기화합니다.
        
        Args:
            epsilon: 0으로 나누기 방지를 위한 작은 값
            shape: 저장할 통계의 형태
        """
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")  # 초기 분산은 1로 설정하여 초기 정규화에 영향 감소
        self.count = epsilon
        self.shape = shape
        self.epsilon = epsilon
        
        # 추가: 최소/최대 허용 분산 설정
        self.var_min = 1e-4
        self.var_max = 1e6

    def update(self, x):
        """
        배치 데이터로 평균과 분산을 업데이트합니다.
        
        Args:
            x: 업데이트할 데이터 배치 (첫 번째 차원이 배치 크기)
        """
        # 입력 데이터 유효성 확인
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # NaN, Inf 처리
        x = np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
            
        # 배치 크기 확인
        if x.shape[0] == 0:
            return
        
        # 형태 확인 및 차원 조정
        if len(self.shape) > 0 and x.shape[1:] != self.shape:
            try:
                x = np.reshape(x, (x.shape[0],) + self.shape)
            except:
                return  # 형태 불일치 시 업데이트 중단
        
        # 기본 통계 계산
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        # 이상치 필터링 (선택적)
        # 분산이 너무 큰 경우 통계치 제한
        batch_var = np.clip(batch_var, self.var_min, self.var_max)
        
        # 모멘트 기반 업데이트
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        계산된 평균과 분산으로 내부 상태를 업데이트합니다.
        
        Args:
            batch_mean: 배치의 평균
            batch_var: 배치의 분산
            batch_count: 배치의 샘플 수
        """
        # 0으로 나누기 방지
        if batch_count <= 0:
            return
            
        # NaN/Inf 검사
        if np.any(np.isnan(batch_mean)) or np.any(np.isnan(batch_var)):
            return
            
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # 새 평균 계산
        new_mean = self.mean + delta * batch_count / tot_count
        
        # 새 분산 계산 (Welford의 알고리즘)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        # 분산 클리핑 (너무 작거나 큰 값 방지)
        new_var = np.clip(new_var, self.var_min, self.var_max)
        
        # 카운터 업데이트
        new_count = tot_count

        # 상태 업데이트
        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        
    def normalize(self, x):
        """
        주어진 입력 x를 현재 통계를 사용해 정규화합니다.
        
        Args:
            x: 정규화할 입력 데이터
            
        Returns:
            정규화된 데이터
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # NaN/Inf 처리
        x = np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
            
        # 정규화 수행 (Z-점수)
        normalized_x = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        
        return normalized_x
        
    def reset(self):
        """
        통계를 초기 상태로 재설정합니다.
        """
        self.mean = np.zeros(self.shape, "float64")
        self.var = np.ones(self.shape, "float64")
        self.count = self.epsilon 