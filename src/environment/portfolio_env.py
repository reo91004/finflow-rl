"""
주식 포트폴리오 강화학습 환경 모듈

주식 포트폴리오 관리를 위한: gymnasium 기반 강화학습 환경을 구현합니다.
이 환경은 다양한 자산에 대한 포트폴리오 배분 전략을 학습하기 위한 기본 프레임워크를 제공합니다.
상태는 각 자산의 기술적 지표, 행동은 자산 배분 비율, 보상은 포트폴리오 가치 변화 기반으로 설계되었습니다.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.models.running_mean_std import RunningMeanStd
from src.constants import (
    INITIAL_CASH, 
    COMMISSION_RATE, 
    MAX_EPISODE_LENGTH, 
    NORMALIZE_STATES,
    DEFAULT_GAMMA, 
    ACTION_PENALTY_COEF,
    REWARD_SHARPE_WINDOW,
    REWARD_RETURN_WEIGHT,
    REWARD_SHARPE_WEIGHT,
    REWARD_DRAWDOWN_PENALTY,
    REWARD_VOL_SCALE_MIN,
    REWARD_VOL_SCALE_MAX,
    REWARD_ACCUMULATION_DAYS,
    REWARD_LONG_TERM_BONUS,
    REWARD_NEGATIVE_WEIGHT,
    REWARD_CLIP_MIN,
    REWARD_CLIP_MAX,
    SHARPE_RATIO_CLIP,
    RMS_EPSILON,
    CLIP_OBS,
    CLIP_REWARD
)

class StockPortfolioEnv(gym.Env):
    """
    주식 포트폴리오 관리를 위한 강화학습 환경입니다.

    - 관측(Observation): 각 자산의 기술적 지표 (10개 피처)
    - 행동(Action): 각 자산에 대한 투자 비중 (0~1 사이 값, 총합 1)
    - 보상(Reward): 포트폴리오 가치의 선형 변화율 (tanh 클리핑 적용)
    - 상태 정규화(State Normalization): RunningMeanStd를 이용한 관측값 및 보상 정규화 기능 포함
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: np.ndarray,
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_RATE,
        max_episode_length=MAX_EPISODE_LENGTH,
        normalize_states=NORMALIZE_STATES,
        gamma=DEFAULT_GAMMA,
        action_penalty_coef=ACTION_PENALTY_COEF,
    ):
        super(StockPortfolioEnv, self).__init__()
        self.data = data  # (n_steps, n_assets, n_features)
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_episode_length = max_episode_length
        self.normalize_states = normalize_states
        self.gamma = gamma  # 보상 정규화 시 사용
        self.action_penalty_coef = action_penalty_coef

        self.n_steps, self.n_assets, self.n_features = data.shape

        # 상태 공간 정의
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets, self.n_features),
            dtype=np.float32,
        )

        # 행동 공간 정의 (무위험 자산(현금) 포함 n_assets + 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # 상태/보상 정규화 객체 초기화
        if self.normalize_states:
            self.obs_rms = RunningMeanStd(shape=(self.n_assets, self.n_features))
            self.ret_rms = RunningMeanStd(shape=())
            self.returns_norm = np.zeros(1)  # 정규화된 누적 보상 추적
        else:
            self.obs_rms = None
            self.ret_rms = None

        # 내부 상태 변수 초기화 (reset에서 수행)
        self.current_step = 0
        self.start_index = 0  # 에피소드 시작 인덱스 저장
        self.cash = 0.0
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)  # 보유 주식 수
        self.portfolio_value = 0.0  # 현재 포트폴리오 가치
        self.weights = np.ones(self.n_assets + 1) / (
            self.n_assets + 1
        )  # 현재 자산 비중 (현금 포함)
        self.prev_weights = np.ones(self.n_assets + 1) / (
            self.n_assets + 1
        )  # 이전 자산 비중 (행동 변화 페널티용)

        # K-일 보상 누적을 위한 가치 이력
        self.portfolio_value_history = []

        # Sharpe ratio 계산을 위한 수익률 이력
        self.returns_history = []

        # 최대 가치 추적 (드로우다운 계산용)
        self.max_portfolio_value = 0.0

        # 시장 변동성 추적 (가변 클리핑용)
        self.market_volatility_window = []
        self.volatility_scaling = 1.0  # 기본값은 1.0 (표준 클리핑)
        
        # 초기화 시 적용할 배치 크기
        self.initial_batch_size = 100

    def _normalize_obs(self, obs):
        """관측값을 정규화합니다."""
        if not self.normalize_states or self.obs_rms is None:
            return obs
        
        # 관측값을 배치 형태로 변환
        obs_batch = obs.reshape(1, self.n_assets, self.n_features)
        
        # 초기화 단계에서는 더 큰 배치로 통계 누적
        if self.current_step < self.initial_batch_size:
            # 초기 통계 구축을 위한 인공 배치 생성
            noise_scale = 0.01  # 적은 노이즈 추가
            artificial_batch = obs + np.random.normal(0, noise_scale, size=obs.shape)
            artificial_batch = artificial_batch.reshape(1, self.n_assets, self.n_features)
            # RunningMeanStd 업데이트
            self.obs_rms.update(artificial_batch)
        
        # 정규화를 위한 통계 업데이트
        if self.current_step % 5 == 0:  # 5스텝마다 업데이트 (갱신 빈도 조절)
            self.obs_rms.update(obs_batch)
        
        # 정규화 및 클리핑
        normalized_obs = np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + RMS_EPSILON),
            -CLIP_OBS,
            CLIP_OBS,
        )
        
        return normalized_obs

    def _normalize_reward(self, reward):
        """보상을 정규화합니다."""
        if not self.normalize_states or self.ret_rms is None:
            # 정규화하지 않는 경우에도 일정 범위로 클리핑
            return np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
            
        # 음수 보상에 대한 가중치 적용
        if reward < 0:
            reward = reward * REWARD_NEGATIVE_WEIGHT
        
        # 이상치 제거 (너무 큰 값이나 너무 작은 값)
        if np.abs(reward) > 2.0:  # 절댓값이 2.0 이상인 보상 조정
            reward = np.sign(reward) * (np.log(1 + np.abs(reward)) + 1.0)
        
        # 누적 할인 보상 업데이트
        self.returns_norm = self.gamma * self.returns_norm + reward
        
        # 보상 정규화 통계 업데이트 (매 스텝마다)
        if self.current_step % 1 == 0:
            self.ret_rms.update(np.array([self.returns_norm]))
        
        # 보상 정규화 및 클리핑
        std = np.sqrt(self.ret_rms.var + RMS_EPSILON)
        normalized_reward = np.clip(
            reward / (std + 1e-8), 
            REWARD_CLIP_MIN, 
            REWARD_CLIP_MAX
        )
        
        return normalized_reward

    def _calculate_sharpe_ratio(self, window_size=REWARD_SHARPE_WINDOW):
        """
        최근 window_size 기간의 Sharpe ratio를 계산합니다.

        Args:
            window_size (int): Sharpe ratio 계산에 사용할 기간 길이.

        Returns:
            float: 계산된 Sharpe ratio 값 (클리핑 적용).
        """
        if len(self.returns_history) < 2:  # 최소 2개의 수익률이 필요
            return 0.0

        # 최근 수익률만 사용
        recent_returns_raw = (
            self.returns_history[-window_size:]
            if len(self.returns_history) >= window_size
            else self.returns_history
        )
        
        # NaN 및 Inf 값 처리
        recent_returns = [r for r in recent_returns_raw if np.isfinite(r)]
        if len(recent_returns) < 2: # 유효한 데이터가 2개 미만이면 0 반환
            return 0.0
            
        # 이상치 제거 (3시그마 룰 적용)
        mean = np.mean(recent_returns)
        std = np.std(recent_returns)
        if std > 0:
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            filtered_returns = [r for r in recent_returns if lower_bound <= r <= upper_bound]
            if len(filtered_returns) >= 2:
                recent_returns = filtered_returns

        # 평균 수익률과 표준편차 계산
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        # 표준편차가 0에 가까우면 Sharpe ratio는 정의되지 않음
        if std_return < 1e-8:
            return 0.0 if mean_return < 0 else SHARPE_RATIO_CLIP  # 양수 수익률이면 최대 클리핑 값

        # 일별 Sharpe ratio 계산 (무위험 수익률 0 가정)
        daily_sharpe = mean_return / std_return

        # 연율화 (252 거래일 기준)
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        
        # Sharpe ratio 클리핑
        clipped_sharpe = np.clip(annualized_sharpe, -SHARPE_RATIO_CLIP, SHARPE_RATIO_CLIP)

        return clipped_sharpe

    def _calculate_drawdown(self):
        """
        현재 드로우다운을 계산합니다.

        Returns:
            float: 현재 드로우다운 값 (0~1 사이, 높을수록 큰 손실)
        """
        if self.max_portfolio_value <= 1e-8:
            return 0.0

        current_drawdown = 1 - (self.portfolio_value / self.max_portfolio_value)
        return max(0.0, current_drawdown)  # 음수가 되지 않도록

    def _calculate_volatility_scaling(self, window_size=20):
        """
        시장 변동성에 기반한 보상 클리핑 스케일링 값을 계산합니다.
        변동성이 높을수록 클리핑이 약해지고, 낮을수록 강해집니다.

        Args:
            window_size (int): 변동성 계산에 사용할 기간 길이.

        Returns:
            float: 보상 클리핑에 사용할 스케일링 값.
        """
        if len(self.market_volatility_window) < 2:
            return 1.0  # 기본값

        # 최근 변동성 데이터만 사용
        recent_volatility = (
            self.market_volatility_window[-window_size:]
            if len(self.market_volatility_window) >= window_size
            else self.market_volatility_window
        )

        # 평균 변동성 계산
        avg_volatility = np.mean(recent_volatility)

        if avg_volatility < 1e-8:
            return REWARD_VOL_SCALE_MIN  # 변동성이 매우 낮은 경우

        # 변동성이 높을수록 클리핑이 약화됨 (스케일 값이 커짐)
        # 참고: market_volatility_window에는 표준편차 값이 저장되어 있다고 가정
        scaling = np.clip(
            avg_volatility / 0.01, REWARD_VOL_SCALE_MIN, REWARD_VOL_SCALE_MAX
        )

        return scaling

    def reset(self, *, seed=None, options=None, start_index=None):
        """환경을 초기 상태로 리셋합니다."""
        super().reset(seed=seed)
        import logging
        logger = logging.getLogger("PortfolioRL")

        # 에피소드 시작 인덱스 설정 (데이터 길이 내 무작위 또는 0)
        if start_index is None:
            max_start_index = max(0, self.n_steps - self.max_episode_length)
            if max_start_index == 0:
                self.start_index = 0
            else:
                self.start_index = np.random.randint(
                    0, max_start_index + 1
                )  # 0부터 시작 가능하도록 +1
        elif start_index >= self.n_steps:
            self.start_index = max(0, self.n_steps - 1)
            logger.warning(f"시작 인덱스가 데이터 길이를 초과하여 {self.start_index}로 조정됨")
        else:
            self.start_index = start_index

        self.current_step = self.start_index # 현재 스텝을 시작 인덱스로 설정
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[0] = 1.0  # 처음에는 전부 현금
        self.prev_weights = self.weights.copy()

        # 이력 비우기
        self.portfolio_value_history = [self.portfolio_value]
        self.returns_history = []
        self.max_portfolio_value = self.portfolio_value
        self.market_volatility_window = []
        self.volatility_scaling = 1.0

        # 정규화 통계 초기화
        if self.normalize_states and self.ret_rms is not None:
            self.returns_norm = np.zeros(1)  # 누적 보상 초기화

        # 초기 관측 반환
        obs = self._get_observation()
        info = self._get_info()

        return obs, info
        
    def _get_observation(self):
        """현재 상태에 대한 관측값을 반환합니다."""
        observation = self.data[self.current_step].copy()
        return self._normalize_obs(observation)

    def _get_info(self):
        """현재 상태에 대한 부가 정보를 반환합니다."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "current_step": self.current_step,
            "max_value": self.max_portfolio_value,
            "drawdown": self._calculate_drawdown(),
            "return": 0.0 if len(self.returns_history) == 0 else self.returns_history[-1],
            "sharpe_ratio": self._calculate_sharpe_ratio(),
        }

    def step(self, action):
        """
        주어진 행동(포트폴리오 비중)에 따라 환경을 한 스텝 진행합니다.

        Args:
            action (np.ndarray): 각 자산의 목표 비중 (n_assets + 1 크기 배열, 합계 1)

        Returns:
            tuple: (관측, 보상, 종료여부, 절단여부, 정보)
        """
        # 유효한 행동인지 확인
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
            
        # 행동이 음수면 0으로 클리핑
        action = np.clip(action, 0.0, 1.0)
            
        # 합이 1이 되도록 정규화
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            # 모든 비중이 0이면 모두 현금으로
            action = np.zeros_like(action)
            action[0] = 1.0

        # 이전 포트폴리오 가치 저장
        prev_portfolio_value = self.portfolio_value

        # 현재 주가 가져오기
        current_prices = self.data[self.current_step, :, 3]  # Close 가격 사용

        # 목표 자산 배분에 따라 거래 실행
        self._execute_trades(action, current_prices)

        # 포트폴리오 이력 추가
        self.portfolio_value_history.append(self.portfolio_value)

        # 다음 스텝으로 이동
        self.current_step += 1

        # 포트폴리오 수익률 계산 및 이상치 처리
        raw_return = (self.portfolio_value / prev_portfolio_value) - 1.0
        
        # 이상치 관리: 너무 큰 변동성은 로그 스케일로 변환하여 완화
        if np.abs(raw_return) > 0.1:  # 10% 이상 변동은 로그 스케일로 변환
            raw_return = np.sign(raw_return) * np.log(1 + np.abs(raw_return))
            
        self.returns_history.append(raw_return)

        # 최대 포트폴리오 가치 업데이트
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
            
        # 가격 변동성 추적 (현재 데이터의 일일 변동성)
        if self.current_step > 0 and self.current_step < len(self.data):
            try:
                yesterday_close = self.data[self.current_step - 1, :, 3]  # 전일 종가
                today_close = self.data[self.current_step, :, 3]  # 당일 종가
                
                # NaN 및 무한값 처리
                valid_mask = np.isfinite(yesterday_close) & np.isfinite(today_close) & (yesterday_close > 0)
                if np.any(valid_mask):
                    returns = (today_close[valid_mask] / yesterday_close[valid_mask]) - 1.0  # 수익률
                    volatility = np.std(returns)  # 변동성 (전체 자산의 표준편차)
                    
                    # 이상치 처리 - 너무 높은 변동성은 제한
                    volatility = min(volatility, 0.05)  # 5% 이상 변동성은 제한
                    
                    self.market_volatility_window.append(volatility)
                    
                    # 변동성에 따른 보상 스케일링 값 업데이트
                    self.volatility_scaling = self._calculate_volatility_scaling()
            except Exception as e:
                # 변동성 계산 중 오류 발생시 기본값 사용
                self.volatility_scaling = 1.0
        
        # 행동 변화 페널티 계산 - 지나치게 잦은 포트폴리오 변화 방지
        action_change_penalty = 0.0
        if self.action_penalty_coef > 0:
            action_diff = np.abs(action - self.prev_weights).mean()
            
            # 작은 변화는 무시, 큰 변화에만 페널티 부여
            if action_diff > 0.1:  # 10% 이상 변화에만 페널티
                action_change_penalty = self.action_penalty_coef * action_diff

        # 기본 보상 계산 (K-일 수익률) - 단기/중기 수익성 평가
        if REWARD_ACCUMULATION_DAYS <= 1:
            # 단일 스텝 보상
            reward_return = raw_return
        else:
            # K-일 누적 보상
            if len(self.portfolio_value_history) > REWARD_ACCUMULATION_DAYS:
                k_day_return = (
                    self.portfolio_value_history[-1] / self.portfolio_value_history[-REWARD_ACCUMULATION_DAYS]
                ) - 1.0
                
                # 중간 수익률도 고려 (지속적 수익성 확인)
                mid_point = max(1, REWARD_ACCUMULATION_DAYS // 2)
                mid_return = (
                    self.portfolio_value_history[-1] / self.portfolio_value_history[-mid_point]
                ) - 1.0
                
                # 단기 및 중기 수익률 모두 고려
                reward_return = 0.7 * k_day_return + 0.3 * mid_return
            else:
                reward_return = raw_return

        # Sharpe ratio 계산 (안정적인 투자를 위한 변동성 고려)
        sharpe = self._calculate_sharpe_ratio()
        
        # 드로우다운 페널티 계산 - 손실 위험 제한
        drawdown = self._calculate_drawdown()
        # 드로우다운에 대한 비선형 페널티 적용 (손실이 커질수록 페널티 가중)
        if drawdown > 0:
            drawdown_penalty = REWARD_DRAWDOWN_PENALTY * (drawdown + 0.5 * drawdown**2)
        else:
            drawdown_penalty = 0.0
        
        # 장기 보상 보너스 (지속적인 개선을 위한 누적 수익률)
        long_term_bonus = 0.0
        if len(self.returns_history) > 20:  # 최소 20일 이상 데이터가 있을 때
            # 과거 20일 수익률의 누적 곱 (기하 평균 수익률)
            returns_array = np.array(self.returns_history[-20:])
            # 이상치 제거: -50% ~ +50% 범위로 제한
            returns_array = np.clip(returns_array, -0.5, 0.5)
            cumulative_return = np.prod(1 + returns_array) - 1
            
            if cumulative_return > 0:
                # 수익률이 높을수록 보너스 증가 (비선형)
                long_term_bonus = REWARD_LONG_TERM_BONUS * (cumulative_return + 0.2 * cumulative_return**2)

        # 보상 합산
        reward = (
            REWARD_RETURN_WEIGHT * reward_return
            + REWARD_SHARPE_WEIGHT * sharpe
            - drawdown_penalty
            - action_change_penalty
            + long_term_bonus
        )
        
        # 보상 정규화
        normalized_reward = self._normalize_reward(reward)
        
        # 변동성에 따른 보상 스케일링 적용 - 변동성 높을 때 보상 민감도 조정
        scaled_reward = normalized_reward * self.volatility_scaling
        
        # 관측값, 정보, 종료 여부 등 획득
        observation = self._get_observation()
        terminated = (self.portfolio_value <= self.initial_cash * 0.2)  # 자산 80% 이상 손실 시 종료 (기존 90%보다 완화)
        truncated = (self.current_step >= min(self.n_steps - 1, self.start_index + self.max_episode_length))
        info = self._get_info()
        
        # 원시 보상값 정보에 추가
        info["raw_reward"] = reward
        info["normalized_reward"] = normalized_reward
        info["final_reward"] = scaled_reward
        info["sharpe"] = sharpe
        info["action_penalty"] = action_change_penalty
        info["drawdown_penalty"] = drawdown_penalty
        info["long_term_bonus"] = long_term_bonus
        
        # 이전 가중치 업데이트
        self.prev_weights = action.copy()
        
        return observation, scaled_reward, terminated, truncated, info

    def _execute_trades(self, target_value_allocation, current_prices):
        """
        목표 자산 배분에 따라 포트폴리오를 재조정합니다.

        Args:
            target_value_allocation (np.ndarray): 목표 자산 비중 (n_assets + 1 크기 배열, 합계 1)
            current_prices (np.ndarray): 현재 주가 (n_assets 크기 배열)

        Returns:
            float: 거래 후 포트폴리오 가치
        """
        total_value = self.cash + np.sum(self.holdings * current_prices)
        self.portfolio_value = total_value

        # 현금 비중
        cash_ratio = target_value_allocation[0]
        target_cash = total_value * cash_ratio

        # 각 자산의 실제 가치
        current_asset_values = self.holdings * current_prices
        
        # 거래 수수료 추적
        total_commission = 0.0

        # 거래 실행 (각 자산별로)
        for i in range(self.n_assets):
            # 목표 자산 가치
            target_value = total_value * target_value_allocation[i + 1]
            
            # 현재 보유 자산 가치
            current_value = current_asset_values[i]
            
            # 거래 필요 여부 확인 (1% 이상 차이가 있을 때만 거래)
            if abs(target_value - current_value) / total_value > 0.01:
                # 목표 자산 수량
                target_quantity = target_value / current_prices[i]
                
                # 거래량 계산
                trade_quantity = target_quantity - self.holdings[i]
                
                if trade_quantity > 0:  # 매수
                    # 수수료 계산 (매수 금액의 일정 비율)
                    commission = trade_quantity * current_prices[i] * self.commission_rate
                    
                    # 사용할 현금 계산 (매수 금액 + 수수료)
                    required_cash = trade_quantity * current_prices[i] + commission
                    
                    # 사용 가능한 현금이 충분한지 확인
                    if required_cash <= self.cash:
                        # 매수 실행
                        self.holdings[i] += trade_quantity
                        self.cash -= required_cash
                        total_commission += commission
                    else:
                        # 현금이 부족하면 가능한 만큼만 매수
                        affordable_quantity = (self.cash / (current_prices[i] * (1 + self.commission_rate)))
                        if affordable_quantity > 0:
                            commission = affordable_quantity * current_prices[i] * self.commission_rate
                            cost = affordable_quantity * current_prices[i] + commission
                            self.holdings[i] += affordable_quantity
                            self.cash -= cost
                            total_commission += commission
                
                elif trade_quantity < 0:  # 매도
                    # 매도 가능한 수량 확인
                    sell_quantity = min(abs(trade_quantity), self.holdings[i])
                    
                    if sell_quantity > 0:
                        # 수수료 계산 (매도 금액의 일정 비율)
                        sell_value = sell_quantity * current_prices[i]
                        commission = sell_value * self.commission_rate
                        
                        # 매도 실행
                        self.holdings[i] -= sell_quantity
                        self.cash += sell_value - commission
                        total_commission += commission

        # 거래 후 포트폴리오 가치 계산
        self.portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        
        # 각 자산의 실제 포트폴리오 내 비중 계산
        self.weights[0] = self.cash / self.portfolio_value  # 현금 비중
        for i in range(self.n_assets):
            self.weights[i + 1] = (self.holdings[i] * current_prices[i]) / self.portfolio_value
        
        return self.portfolio_value

    def render(self, mode="human"):
        """환경 상태를 시각화합니다."""
        if mode != "human":
            raise NotImplementedError(f"mode {mode}는 지원되지 않습니다.")
            
        info = self._get_info()
        # 실제 렌더링 코드는 별도로 구현해야 함
        return info

    def close(self):
        """환경 종료 시 사용되며, 필요한 자원을 해제합니다."""
        pass 