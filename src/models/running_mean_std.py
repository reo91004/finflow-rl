"""
이동 평균과 표준편차 계산 모듈

Welford의 알고리즘을 사용하여 상태 및 보상의 정규화를 위한 이동 평균과 표준편차를 계산하는 클래스를 구현합니다.
주로 강화학습 환경에서 입력 상태나 보상의 스케일을 정규화하기 위해 사용됩니다.
"""

import numpy as np
from src.constants import RMS_EPSILON, CLIP_OBS

class RunningMeanStd:
    """
    Welford's online algorithm을 사용하여 이동 평균과 표준편차를 계산합니다.
    상태 및 보상 정규화에 사용됩니다.
    Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, epsilon=RMS_EPSILON, shape=(), clip_range=CLIP_OBS):
        """
        이동 평균 및 표준편차 계산기를 초기화합니다.
        
        Args:
            epsilon: 0으로 나누기 방지를 위한 작은 값
            shape: 저장할 통계의 형태
            clip_range: 정규화된 값의 클리핑 범위
        """
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")  # 초기 분산은 1로 설정하여 초기 정규화에 영향 감소
        self.count = epsilon
        self.shape = shape
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # 추가: 최소/최대 허용 분산 설정
        self.var_min = 1e-6  # 더 낮은 최소 분산 (1e-4→1e-6)
        self.var_max = 1e4   # 더 낮은 최대 분산 (1e6→1e4)
        
        # 추가: 평균 업데이트 시 사용할 적응형 가중치
        self.update_weight = 0.005  # 초기 업데이트 가중치
        self.min_update_weight = 0.001  # 최소 업데이트 가중치

    def update(self, x):
        """
        배치 데이터로 평균과 분산을 업데이트합니다.
        
        Args:
            x: 업데이트할 데이터 배치 (첫 번째 차원이 배치 크기)
        """
        # 입력 데이터 유효성 확인
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # NaN, Inf 처리 - 더 보수적인 값으로 대체
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            
        # 배치 크기 확인
        if x.shape[0] == 0:
            return
        
        # 형태 확인 및 차원 조정
        if len(self.shape) > 0 and x.shape[1:] != self.shape:
            try:
                x = np.reshape(x, (x.shape[0],) + self.shape)
            except:
                return  # 형태 불일치 시 업데이트 중단
        
        # 이상치 제거: 너무 극단적인 값 필터링 (선택적 기능)
        if x.shape[0] > 10:  # 충분한 샘플이 있을 때만 적용
            # 평균으로부터 5 표준편차 이상 벗어난 샘플 제외
            stddev = np.sqrt(self.var + self.epsilon)
            z_scores = np.abs((x - self.mean) / stddev)
            mask = z_scores < 5.0  # 5 시그마 규칙
            
            # 차원에 따라 마스크 적용
            if len(x.shape) > 1:
                # 첫 번째 차원에 대해 마스크 집계
                mask = np.all(mask, axis=tuple(range(1, len(x.shape))))
                x = x[mask]
            
            # 이상치 제거 후 배치가 비어있으면 반환
            if x.shape[0] == 0:
                return
        
        # 기본 통계 계산
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        # 배치 분산 클리핑
        batch_var = np.clip(batch_var, self.var_min, self.var_max)
        
        # 가중치 적용 업데이트 - 시간에 따라 점점 느리게 적응하도록 설정
        self.update_with_weight(batch_mean, batch_var, batch_count)

    def update_with_weight(self, batch_mean, batch_var, batch_count):
        """
        적응형 가중치를 사용하여 내부 상태를 업데이트합니다.
        
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
        
        # 샘플 수에 따른 적응형 가중치 계산
        sample_weight = min(self.update_weight, 1.0 / (1.0 + self.count / 1000.0))
        sample_weight = max(sample_weight, self.min_update_weight)
        
        # 평균 업데이트
        self.mean = (1.0 - sample_weight) * self.mean + sample_weight * batch_mean
        
        # 분산 업데이트 (Welford 알고리즘 기반 공식)
        self.var = (1.0 - sample_weight) * self.var + sample_weight * (
            batch_var + np.square(batch_mean - self.mean)
        )
        
        # 분산 클리핑
        self.var = np.clip(self.var, self.var_min, self.var_max)
        
        # 카운터 업데이트
        self.count += batch_count * sample_weight

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
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            
        # 정규화 수행 (Z-점수)
        normalized_x = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        
        # 정규화된 값 클리핑
        normalized_x = np.clip(normalized_x, -self.clip_range, self.clip_range)
        
        return normalized_x
        
    def reset(self):
        """
        통계를 초기 상태로 재설정합니다.
        """
        self.mean = np.zeros(self.shape, "float64")
        self.var = np.ones(self.shape, "float64")
        self.count = self.epsilon
        
    def get_stats(self):
        """
        현재 통계 정보를 반환합니다.
        
        Returns:
            dict: 현재 평균, 분산, 표준편차와 샘플 수
        """
        return {
            "mean": self.mean,
            "var": self.var,
            "std": np.sqrt(self.var),
            "count": self.count
        } 