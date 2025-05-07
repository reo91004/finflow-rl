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

    def _normalize_obs(self, obs):
        """관측값을 정규화합니다."""
        if not self.normalize_states or self.obs_rms is None:
            return obs
        # RunningMeanStd 업데이트 (차원 맞추기)
        self.obs_rms.update(obs.reshape(1, self.n_assets, self.n_features))
        # 정규화 및 클리핑
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + RMS_EPSILON),
            -CLIP_OBS,
            CLIP_OBS,
        )

    def _normalize_reward(self, reward):
        """보상을 정규화합니다."""
        if not self.normalize_states or self.ret_rms is None:
            return reward
        # 누적 할인 보상 업데이트
        self.returns_norm = self.gamma * self.returns_norm + reward
        # RunningMeanStd 업데이트
        self.ret_rms.update(self.returns_norm)
        # 정규화 및 클리핑
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + RMS_EPSILON), -CLIP_REWARD, CLIP_REWARD
        )

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
        recent_returns = (
            self.returns_history[-window_size:]
            if len(self.returns_history) >= window_size
            else self.returns_history
        )

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
        
        # Sharpe ratio 클리핑 (3.0 → 2.0으로 감소)
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
                start_index = 0
            else:
                start_index = np.random.randint(
                    0, max_start_index + 1
                )  # 0부터 시작 가능하도록 +1
        elif start_index >= self.n_steps:
            logger.warning(
                f"제공된 시작 인덱스({start_index})가 데이터 범위({self.n_steps})를 벗어남. 0으로 설정."
            )
            start_index = 0

        self.current_step = start_index

        # 내부 상태 초기화
        self.cash = self.initial_cash
        self.holdings.fill(0)
        self.portfolio_value = self.cash
        self.weights = np.ones(self.n_assets + 1) / (
            self.n_assets + 1
        )  # 현금 포함 균등 비중
        self.prev_weights = self.weights.copy()

        # 보상 정규화 관련 변수 초기화
        if self.normalize_states:
            self.returns_norm = np.zeros(1)

        # K-일 보상 누적 이력 초기화
        self.portfolio_value_history = [self.portfolio_value]

        # 수익률 이력 초기화
        self.returns_history = []

        # 최대 가치 초기화
        self.max_portfolio_value = self.portfolio_value

        # 시장 변동성 초기화
        self.market_volatility_window = []
        self.volatility_scaling = 1.0

        # 초기 관측값 반환
        observation = self._get_observation()
        normalized_observation = self._normalize_obs(observation)
        info = self._get_info()  # 초기 정보 생성

        return normalized_observation.astype(np.float32), info

    def _get_observation(self):
        """현재 스텝의 원본 관측 데이터를 반환합니다."""
        # 데이터 인덱스 범위 확인 (방어 코드)
        step = min(self.current_step, self.n_steps - 1)  # 범위를 벗어나지 않도록 조정
        return self.data[step]

    def _get_info(self):
        """현재 환경 상태 정보를 담은 딕셔너리를 반환합니다."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "return": 0.0,  # 초기 상태에서는 수익률 0
            "raw_reward": 0.0,  # 초기 상태에서는 보상 0
        }

    def step(self, action):
        """
        환경을 한 스텝 진행시킵니다.

        Args:
            action (np.ndarray): 에이전트가 선택한 행동 (자산별 목표 비중, 현금 포함).

        Returns:
            tuple: (next_observation, reward, terminated, truncated, info)
                   - next_observation (np.ndarray): 정규화된 다음 상태 관측값.
                   - reward (float): 정규화된 보상.
                   - terminated (bool): 에피소드 종료 여부 (파산 또는 데이터 끝 도달).
                   - truncated (bool): 에피소드 중단 여부 (최대 길이 도달).
                   - info (dict): 추가 정보 (포트폴리오 가치, 현금, 수익률 등).
        """
        import logging
        logger = logging.getLogger("PortfolioRL")
        # 행동 크기 확인 및 조정 (현금 슬롯 추가)
        if len(action) == self.n_assets:  # 기존 호환성 유지
            action_with_cash = np.append(action, 0.0)  # 현금 비중 0 추가
        else:
            action_with_cash = action

        # 행동 정규화 (비중 합 1)
        action_with_cash = np.clip(action_with_cash, 0, 1)
        action_sum = action_with_cash.sum()
        if action_sum > 1e-6:
            action_with_cash = action_with_cash / action_sum
        else:  # 비중 합이 0에 가까우면 균등 분배
            action_with_cash = np.ones(self.n_assets + 1) / (self.n_assets + 1)

        # 현재 가격 정보 (원본 데이터 사용)
        current_obs = self._get_observation()
        current_prices = np.maximum(current_obs[:, 3], 1e-6)  # 종가, 0 방지

        # 이전 포트폴리오 가치
        prev_portfolio_value = self.cash + np.dot(self.holdings, current_prices)
        self.portfolio_value_history.append(prev_portfolio_value)

        # 최대 포트폴리오 가치 업데이트
        self.max_portfolio_value = max(self.max_portfolio_value, prev_portfolio_value)

        # 최근 K일 가치만 유지
        if len(self.portfolio_value_history) > REWARD_ACCUMULATION_DAYS + 1:
            self.portfolio_value_history.pop(0)

        # 파산 조건 확인
        if prev_portfolio_value <= 1e-6:
            terminated = True
            truncated = False
            raw_reward = -10.0  # 파산 시 큰 음수 보상
            info = {
                "portfolio_value": 0.0,
                "cash": 0.0,
                "holdings": self.holdings.copy(),
                "weights": np.zeros_like(self.weights),
                "return": -1.0,
                "raw_reward": raw_reward,
            }
            # 마지막 관측값은 현재 관측값 사용 (정규화)
            last_obs_norm = self._normalize_obs(current_obs)
            reward_norm = self._normalize_reward(raw_reward)
            return (
                last_obs_norm.astype(np.float32),
                float(reward_norm),
                terminated,
                truncated,
                info,
            )

        # 행동 변화에 따른 페널티 계산 (L1 거리, 페널티 강화)
        # 작은 변화는 무시하고, 큰 변화에 대해서만 페널티 적용 (실제 거래에 맞게)
        weight_changes = np.abs(action_with_cash - self.prev_weights)
        
        # 임계값 이하의 작은 변화는 무시
        threshold = 0.02  # 2% 이하의 변화는 무시
        significant_changes = np.where(weight_changes > threshold, weight_changes, 0.0)
        
        # 페널티 계수 적용
        action_change_penalty = self.action_penalty_coef * np.sum(significant_changes)

        # 목표 자산 가치 계산 (현금 제외한 주식 부분만)
        stock_weights = action_with_cash[:-1]
        cash_weight = action_with_cash[-1]
        target_value_allocation = stock_weights * prev_portfolio_value

        # 실제 거래 실행 (매수/매도)
        self._execute_trades(target_value_allocation, current_prices)

        # 현금 비중에 따라 현금 조정 (위 거래 후에 남은 현금 비율 조정)
        target_cash = cash_weight * prev_portfolio_value
        # 현재 현금이 목표보다 많으면 유지, 적으면 다른 자산 비중 줄여서 조정은 skip

        # 다음 스텝으로 이동
        self.current_step += 1
        terminated = self.current_step >= self.n_steps  # 종료 조건: 마지막 스텝 이후
        truncated = False  # Truncated는 학습 루프에서 제어

        # 다음 스텝 가격 및 새 포트폴리오 가치 계산
        if terminated:
            next_obs_raw = current_obs  # 마지막 스텝이면 현재 관측값 사용
        else:
            next_obs_raw = self._get_observation()  # _get_observation 사용

        next_prices = np.maximum(next_obs_raw[:, 3], 1e-6)  # 다음 날 종가, 0 방지
        self.portfolio_value = self.cash + np.dot(self.holdings, next_prices)

        # 가중치 업데이트 (0으로 나누기 방지), 현금 포함
        if self.portfolio_value > 1e-8:
            stock_weights = (
                self.holdings * next_prices
            ) / self.portfolio_value  # 주식 비중
            cash_weight = self.cash / self.portfolio_value  # 현금 비중
            self.weights = np.append(stock_weights, cash_weight)
        else:
            self.weights = np.zeros(self.n_assets + 1)

        # 이전 가중치 저장 (다음 스텝의 변화 페널티 계산용)
        self.prev_weights = self.weights.copy()

        # 수익률 계산 및 이력 업데이트
        prev_value_safe = max(
            prev_portfolio_value, 1e-8
        )  # 이전 가치가 0에 가까울 때 대비
        current_value_safe = max(
            self.portfolio_value, 0.0
        )  # 현재 가치는 0이 될 수 있음
        daily_return = (current_value_safe / prev_value_safe) - 1
        self.returns_history.append(daily_return)

        # 시장 변동성 업데이트 (최근 수익률의 표준편차)
        if len(self.returns_history) > 1:
            window_size = min(20, len(self.returns_history))
            recent_vol = np.std(self.returns_history[-window_size:])
            self.market_volatility_window.append(recent_vol)
            # 최근 변동성만 저장
            if len(self.market_volatility_window) > 100:  # 충분히 긴 이력 유지
                self.market_volatility_window.pop(0)

            # 변동성 기반 클리핑 스케일 업데이트
            self.volatility_scaling = self._calculate_volatility_scaling()

        # --- 개선된 보상 계산 방식 ---

        # 1. K-일 누적 수익률 계산 (장기적 성과 고려)
        if len(self.portfolio_value_history) > REWARD_ACCUMULATION_DAYS:
            k_day_ago_value = self.portfolio_value_history[-REWARD_ACCUMULATION_DAYS-1]
            if k_day_ago_value > 1e-8:
                k_day_return = (current_value_safe / k_day_ago_value) - 1
            else:
                k_day_return = -1.0
        else:
            k_day_return = daily_return
        
        # 장기 보상 보너스 적용 (0.3 → 0.1로 감소)
        long_term_bonus = REWARD_LONG_TERM_BONUS * k_day_return if k_day_return > 0 else 0

        # 2. Sharpe ratio 계산 (개선됨)
        sharpe_ratio = self._calculate_sharpe_ratio()
        # Sharpe ratio 가중치 조정
        sharpe_component = sharpe_ratio * REWARD_SHARPE_WEIGHT

        # 3. 드로우다운 페널티
        drawdown = self._calculate_drawdown()
        drawdown_penalty = REWARD_DRAWDOWN_PENALTY * drawdown

        # 4. 최종 보상 계산 (개선된 가중치 및 구성 요소)
        # 일일 수익률 (주요 보상 구성 요소)
        return_component = np.tanh(daily_return) * REWARD_RETURN_WEIGHT
        
        # 음수 보상에 더 높은 가중치 적용 (1.2 → 1.1로 감소)
        if return_component < 0:
            return_component *= REWARD_NEGATIVE_WEIGHT

        # 최종 보상 계산 (모든 구성 요소 합산)
        raw_reward = return_component + sharpe_component + long_term_bonus - drawdown_penalty - action_change_penalty

        # NaN/Inf 처리
        if np.isnan(raw_reward) or np.isinf(raw_reward):
            raw_reward = -1.0
            
        # 보상 클리핑 적용 (-5.0 ~ 5.0)
        raw_reward = np.clip(raw_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)

        # 다음 상태 및 보상 정규화
        next_obs_norm = self._normalize_obs(next_obs_raw)
        reward_norm = self._normalize_reward(raw_reward)

        # 정보 업데이트
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "return": daily_return,
            "raw_reward": raw_reward,
            "action_penalty": action_change_penalty,
            "k_day_return": k_day_return if "k_day_return" in locals() else 0.0,
            "long_term_bonus": long_term_bonus if "long_term_bonus" in locals() else 0.0,
            "sharpe_ratio": sharpe_ratio,
            "drawdown": drawdown,
            "volatility_scaling": self.volatility_scaling,
        }

        return (
            next_obs_norm.astype(np.float32),
            float(reward_norm),
            terminated,
            truncated,
            info,
        )

    def _execute_trades(self, target_value_allocation, current_prices):
        """
        목표 가치 배분에 따라 포트폴리오를 리밸런싱합니다.
        매수/매도를 실행하고 현금 잔고를 업데이트합니다.

        Args:
            target_value_allocation (np.ndarray): 각 자산별 목표 가치 배분
            current_prices (np.ndarray): 현재 자산 가격
        """
        # 현재 보유량 기반 가치
        current_values = self.holdings * current_prices

        # 리밸런싱 임계값 (포트폴리오 가치의 일정 비율)
        # 이 임계값보다 작은 가치 변화는 리밸런싱하지 않음 (불필요한 거래 감소)
        portfolio_value = self.cash + np.sum(current_values)
        rebalance_threshold = 0.01 * portfolio_value  # 포트폴리오 가치의 1%

        # 매도부터 실행 (현금 확보)
        for i in range(self.n_assets):
            delta_value = target_value_allocation[i] - current_values[i]
            if delta_value < -rebalance_threshold:  # 임계값보다 크게 매도할 경우만 실행
                # 매도할 수량 계산 (소수점 이하 잘라내기)
                sell_amount = np.floor(abs(delta_value) / current_prices[i])
                
                # 실제 매도 가능한 수량은 보유량을 초과할 수 없음
                sell_amount = min(sell_amount, self.holdings[i])
                
                # 매도 실행 및 현금 증가
                if sell_amount > 0:  # 0보다 큰 수량만 매도
                    # 수수료 계산: 매도 금액의 일정 비율
                    sell_value = sell_amount * current_prices[i]
                    commission = sell_value * self.commission_rate
                    
                    # 실제 매도 금액 (수수료 차감)
                    actual_sell_value = sell_value - commission
                    
                    # 현금 및 보유량 업데이트
                    self.cash += actual_sell_value
                    self.holdings[i] -= sell_amount

        # 매수 실행
        for i in range(self.n_assets):
            delta_value = target_value_allocation[i] - current_values[i]
            if delta_value > rebalance_threshold:  # 임계값보다 크게 매수할 경우만 실행
                # 현재 보유 현금으로 매수 가능한 금액 계산 (모든 자산에 공평하게 분배)
                max_buy_value = min(delta_value, self.cash)
                
                # 수수료를 고려한 실제 매수 가능 금액
                # 매수 금액 = 총금액 / (1 + 수수료율)
                actual_buy_value = max_buy_value / (1 + self.commission_rate)
                
                # 매수 수량 계산 (소수점 이하 잘라내기, 소수주 매수 불가)
                buy_amount = np.floor(actual_buy_value / current_prices[i])
                
                if buy_amount > 0:  # 0보다 큰 수량만 매수
                    # 수수료 포함 총 매수 비용
                    total_cost = buy_amount * current_prices[i] * (1 + self.commission_rate)
                    
                    # 현금 및 보유량 업데이트
                    self.cash -= total_cost
                    self.holdings[i] += buy_amount

        # 숫자 정밀도 오류 방지 (현금이 음수가 되지 않도록)
        self.cash = max(0.0, self.cash)

    def render(self, mode="human"):
        """(선택적) 환경 상태를 간단히 출력합니다."""
        obs = self._get_observation()
        current_prices = obs[:, 3]
        print(f"스텝: {self.current_step}")
        print(f"현금: {self.cash:.2f}")
        print(f"주식 평가액: {np.dot(self.holdings, current_prices):.2f}")
        print(f"총 포트폴리오 가치: {self.portfolio_value:.2f}")

    def close(self):
        """환경 관련 리소스를 정리합니다 (현재는 불필요)."""
        pass 