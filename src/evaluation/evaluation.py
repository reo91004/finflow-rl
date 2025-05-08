"""
포트폴리오 성능 평가 모듈

학습된 모델의 성능을 평가하는 기능을 제공합니다.
백테스팅, 성능 지표 계산, 벤치마크 비교 등의 함수를 포함합니다.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import json
import torch
from tabulate import tabulate

from src.environment.portfolio_env import StockPortfolioEnv
from src.constants import (
    STOCK_TICKERS,
    INITIAL_CASH, 
    BENCHMARK_TICKERS, 
    USE_BENCHMARK,
    DEVICE,
    RESULTS_BASE_PATH
)

def backtest(
    ppo_agent, 
    test_data, 
    test_dates, 
    benchmark_data=None, 
    log_dir=None, 
    logger=None,
    use_ema=True
):
    """
    학습된 에이전트로 백테스팅을 수행합니다.
    
    Args:
        ppo_agent: 평가할 학습된 PPO 에이전트
        test_data: 테스트 데이터 배열 (시간 x 자산 x 피처)
        test_dates: 테스트 날짜 인덱스
        benchmark_data: 벤치마크 데이터 (선택적)
        log_dir: 결과 저장 디렉토리
        logger: 로깅 객체
        use_ema: EMA 모델 사용 여부
        
    Returns:
        dict: 백테스팅 결과 (포트폴리오 가치, 수익률, 성능 지표 등)
    """
    import logging
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(RESULTS_BASE_PATH, f"backtest_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
    
    logger.info(f"백테스팅 시작 (날짜: {test_dates[0]} ~ {test_dates[-1]})")
    
    # 테스트 환경 생성
    env = StockPortfolioEnv(
        data=test_data,
        initial_cash=INITIAL_CASH
    )
    
    # 초기 상태 설정 (시작 인덱스 0)
    state, _ = env.reset(start_index=0)
    
    # 결과 추적 변수 초기화
    days = len(test_dates)
    portfolio_values = np.zeros(days)
    portfolio_weights = np.zeros((days, env.n_assets + 1))  # +1은 현금
    actions_history = []
    returns = np.zeros(days)
    
    # 초기 포트폴리오 가치 기록
    portfolio_values[0] = env.portfolio_value
    portfolio_weights[0] = env.weights
    
    # 인덱스 카운터 초기화
    day_index = 0
    
    # 백테스팅 루프
    done = False
    while not done and day_index < days - 1:
        # 에이전트로 액션 선택
        if hasattr(ppo_agent, 'select_action'):
            action = ppo_agent.select_action(state, use_ema=use_ema)
        else:
            action, _, _ = ppo_agent.policy_old.act(state)
        
        # 액션 실행
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 결과 기록
        day_index += 1
        
        if day_index < days:
            portfolio_values[day_index] = info["portfolio_value"]
            portfolio_weights[day_index] = info["weights"]
            returns[day_index] = info["return"]
            actions_history.append(action)
        
        # 다음 상태로 업데이트
        state = next_state
        
        # 종료 조건 체크
        done = terminated or truncated
    
    # 포트폴리오 가치의 일별 수익률 계산
    daily_returns = np.zeros_like(portfolio_values)
    daily_returns[1:] = (portfolio_values[1:] / portfolio_values[:-1]) - 1
    
    # 누적 수익률 계산
    cumulative_returns = np.cumprod(1 + daily_returns) - 1
    
    # 성능 지표 계산
    metrics = calculate_performance_metrics(daily_returns)
    
    # 결과 저장
    result = {
        "portfolio_values": portfolio_values,
        "portfolio_weights": portfolio_weights,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
        "metrics": metrics,
        "test_dates": test_dates,
        "actions": actions_history if actions_history else None,
    }
    
    # 벤치마크 성능 계산 (있는 경우)
    if USE_BENCHMARK and benchmark_data:
        from src.utils.data_utils import calculate_benchmark_performance
        benchmark_results = calculate_benchmark_performance(benchmark_data, test_dates)
        result["benchmark"] = benchmark_results
        
        # 백테스팅 결과 시각화 및 저장
        plot_backtest_results(
            result, benchmark_results, save_path=os.path.join(log_dir, "backtest_results.png")
        )
        
        # 성능 지표 테이블 생성 및 저장
        create_performance_table(result, benchmark_results, save_path=os.path.join(log_dir, "performance_metrics.txt"))
    else:
        # 벤치마크 없이 시각화
        plot_backtest_results(
            result, None, save_path=os.path.join(log_dir, "backtest_results.png")
        )
    
    # 자산 배분 시각화
    plot_asset_allocation(
        portfolio_weights, test_dates, STOCK_TICKERS, 
        save_path=os.path.join(log_dir, "asset_allocation.png")
    )
    
    # 결과 파일로 저장
    try:
        with open(os.path.join(log_dir, "backtest_results.pkl"), "wb") as f:
            pickle.dump(result, f)
            
        # 주요 성능 지표만 JSON으로도 저장
        with open(os.path.join(log_dir, "performance_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"백테스팅 결과 저장 완료: {log_dir}")
    except Exception as e:
        logger.error(f"결과 저장 중 오류: {e}")
    
    return result

def calculate_performance_metrics(returns):
    """
    포트폴리오 성능 지표를 계산합니다.
    
    Args:
        returns (np.ndarray): 일별 수익률 배열
        
    Returns:
        dict: 계산된 성능 지표들
    """
    # 기본 통계치
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    daily_mean_return = np.mean(returns)
    
    # 변동성
    daily_volatility = np.std(returns)
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # 수익/위험 비율
    sharpe_ratio = (daily_mean_return / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0
    sortino_denominator = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    sortino_ratio = (daily_mean_return / sortino_denominator) * np.sqrt(252) if sortino_denominator > 0 else 0
    
    # 최대 낙폭
    cumulative_returns = np.cumprod(1 + returns) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (1 + peak)
    max_drawdown = np.max(drawdown)
    
    # 승률과 평균 수익
    win_rate = len(returns[returns > 0]) / len(returns)
    gain_loss_ratio = np.abs(np.mean(returns[returns > 0]) / np.mean(returns[returns < 0])) if np.any(returns < 0) else float('inf')
    
    # 상승/하락 연속일수
    positive_streak = get_longest_streak(returns > 0)
    negative_streak = get_longest_streak(returns < 0)
    
    # 월별 수익률 (마지막 252일만 사용)
    monthly_returns = []
    if len(returns) >= 21:
        for i in range(0, min(252, len(returns)), 21):
            if i + 21 <= len(returns):
                monthly_return = np.prod(1 + returns[i:i+21]) - 1
                monthly_returns.append(monthly_return)
    
    # Calmar, Omega, Information Ratio 계산 (고급 지표)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    # 결과 딕셔너리
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
        "win_rate": float(win_rate),
        "gain_loss_ratio": float(gain_loss_ratio),
        "positive_streak": int(positive_streak),
        "negative_streak": int(negative_streak),
        "monthly_returns": [float(r) for r in monthly_returns] if monthly_returns else [],
    }

def get_longest_streak(bool_array):
    """
    불리언 배열에서 연속된 True의 최장 길이를 반환합니다.
    
    Args:
        bool_array (np.ndarray): 불리언 배열
        
    Returns:
        int: 최장 연속 길이
    """
    streak_lengths = []
    current_streak = 0
    
    for value in bool_array:
        if value:
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
                current_streak = 0
    
    if current_streak > 0:
        streak_lengths.append(current_streak)
    
    return max(streak_lengths) if streak_lengths else 0

def ensemble_backtest(
    ensemble_agents, 
    test_data, 
    test_dates, 
    benchmark_data=None, 
    log_dir=None, 
    logger=None,
    use_ema=True,
    voting_method='mean'
):
    """
    앙상블 에이전트로 백테스팅을 수행합니다.
    
    Args:
        ensemble_agents: 앙상블 구성 에이전트 리스트
        test_data: 테스트 데이터
        test_dates: 테스트 날짜 인덱스
        benchmark_data: 벤치마크 데이터
        log_dir: 결과 저장 디렉토리
        logger: 로깅 객체
        use_ema: EMA 모델 사용 여부
        voting_method: 앙상블 투표 방식 ('mean', 'median', 'rank')
        
    Returns:
        dict: 백테스팅 결과
    """
    import logging
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(RESULTS_BASE_PATH, f"ensemble_backtest_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
    
    logger.info(f"앙상블 백테스팅 시작 ({len(ensemble_agents)}개 모델, 방식: {voting_method})")
    
    # 테스트 환경 생성
    env = StockPortfolioEnv(
        data=test_data,
        initial_cash=INITIAL_CASH
    )
    
    # 초기 상태 설정
    state, _ = env.reset(start_index=0)
    
    # 결과 추적 변수 초기화
    days = len(test_dates)
    portfolio_values = np.zeros(days)
    portfolio_weights = np.zeros((days, env.n_assets + 1))  # +1은 현금
    individual_actions = []
    ensemble_actions = []
    returns = np.zeros(days)
    
    # 초기 포트폴리오 가치 기록
    portfolio_values[0] = env.portfolio_value
    portfolio_weights[0] = env.weights
    
    # 인덱스 카운터 초기화
    day_index = 0
    
    # 백테스팅 루프
    done = False
    while not done and day_index < days - 1:
        # 각 에이전트에서 액션 수집
        actions = []
        for agent in ensemble_agents:
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, use_ema=use_ema)
            else:
                action, _, _ = agent.policy_old.act(state)
            actions.append(action)
        
        # 액션 앙상블
        actions_array = np.array(actions)
        if voting_method == 'mean':
            ensemble_action = np.mean(actions_array, axis=0)
        elif voting_method == 'median':
            ensemble_action = np.median(actions_array, axis=0)
        elif voting_method == 'rank':
            # 자산별 순위 투표
            ensemble_action = np.zeros_like(actions_array[0])
            for i in range(len(ensemble_action)):
                ranks = np.argsort(actions_array[:, i])
                weights = np.arange(1, len(ranks) + 1) / np.sum(np.arange(1, len(ranks) + 1))
                for j, rank in enumerate(ranks):
                    ensemble_action[i] += actions_array[rank, i] * weights[j]
        else:
            ensemble_action = np.mean(actions_array, axis=0)  # 기본값
        
        # 합이 1이 되도록 정규화
        ensemble_action = ensemble_action / np.sum(ensemble_action)
        
        # 액션 실행
        next_state, reward, terminated, truncated, info = env.step(ensemble_action)
        
        # 결과 기록
        day_index += 1
        
        if day_index < days:
            portfolio_values[day_index] = info["portfolio_value"]
            portfolio_weights[day_index] = info["weights"]
            returns[day_index] = info["return"]
            individual_actions.append(actions)
            ensemble_actions.append(ensemble_action)
        
        # 다음 상태로 업데이트
        state = next_state
        
        # 종료 조건 체크
        done = terminated or truncated
    
    # 포트폴리오 가치의 일별 수익률 계산
    daily_returns = np.zeros_like(portfolio_values)
    daily_returns[1:] = (portfolio_values[1:] / portfolio_values[:-1]) - 1
    
    # 누적 수익률 계산
    cumulative_returns = np.cumprod(1 + daily_returns) - 1
    
    # 성능 지표 계산
    metrics = calculate_performance_metrics(daily_returns)
    
    # 결과 저장
    result = {
        "portfolio_values": portfolio_values,
        "portfolio_weights": portfolio_weights,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
        "metrics": metrics,
        "test_dates": test_dates,
        "ensemble_actions": ensemble_actions if ensemble_actions else None,
        "individual_actions": individual_actions if individual_actions else None,
        "voting_method": voting_method,
    }
    
    # 벤치마크 성능 계산 (있는 경우)
    if USE_BENCHMARK and benchmark_data:
        from src.utils.data_utils import calculate_benchmark_performance
        benchmark_results = calculate_benchmark_performance(benchmark_data, test_dates)
        result["benchmark"] = benchmark_results
        
        # 백테스팅 결과 시각화 및 저장
        plot_backtest_results(
            result, benchmark_results, 
            save_path=os.path.join(log_dir, "ensemble_backtest_results.png"),
            title=f"Ensemble backtest results (method: {voting_method}, {len(ensemble_agents)} models)"
        )
        
        # 성능 지표 테이블 생성 및 저장
        create_performance_table(
            result, benchmark_results, 
            save_path=os.path.join(log_dir, "ensemble_performance_metrics.txt"),
            title=f"Ensemble performance metrics (method: {voting_method})"
        )
    else:
        # 벤치마크 없이 시각화
        plot_backtest_results(
            result, None, 
            save_path=os.path.join(log_dir, "ensemble_backtest_results.png"),
            title=f"Ensemble backtest results (method: {voting_method}, {len(ensemble_agents)} models)"
        )
    
    # 자산 배분 시각화
    plot_asset_allocation(
        portfolio_weights, test_dates, STOCK_TICKERS, 
        save_path=os.path.join(log_dir, "ensemble_asset_allocation.png"),
        title=f"Ensemble asset allocation (method: {voting_method})"
    )
    
    # 에이전트 간 행동 일치도 시각화 (선택적)
    if individual_actions:
        plot_agent_agreement(
            individual_actions, test_dates, 
            save_path=os.path.join(log_dir, "agent_agreement.png")
        )
    
    # 결과 파일로 저장
    try:
        with open(os.path.join(log_dir, "ensemble_backtest_results.pkl"), "wb") as f:
            pickle.dump(result, f)
            
        # 주요 성능 지표만 JSON으로도 저장
        with open(os.path.join(log_dir, "ensemble_performance_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"앙상블 백테스팅 결과 저장: {log_dir}")
    except Exception as e:
        logger.error(f"결과 저장 중 오류: {e}")
    
    return result

def create_performance_table(result, benchmark_results=None, save_path=None, title="Performance Metrics Comparison"):
    """
    백테스팅 결과에서 성능 지표 테이블을 생성합니다.
    
    Args:
        result: 백테스팅 결과
        benchmark_results: 벤치마크 결과 (선택적)
        save_path: 저장 경로
        title: 테이블 제목
        
    Returns:
        str: 포맷팅된 테이블 문자열
    """
    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError("성능 테이블 생성을 위해 'tabulate' 패키지가 필요합니다. 'pip install tabulate'를 실행하세요.")
    
    # 헤더 준비
    headers = ["지표", "포트폴리오"]
    
    # 벤치마크 티커를 헤더에 미리 추가 (중복 방지)
    if benchmark_results:
        for ticker in benchmark_results.keys():
            headers.append(ticker)
    
    # 행 준비
    rows = []
    
    # 측정 지표 및 단위 정의
    metrics_info = [
        ("총 수익률", result["metrics"]["total_return"], "%"),
        ("연간 수익률", result["metrics"]["annualized_return"], "%"),
        ("샤프 비율", result["metrics"]["sharpe_ratio"], ""),
        ("소르티노 비율", result["metrics"]["sortino_ratio"], ""),
        ("최대 낙폭", result["metrics"]["max_drawdown"], "%"),
        ("변동성", result["metrics"]["annualized_volatility"], "%"),
        ("승률", result["metrics"]["win_rate"], "%"),
        ("손익비", result["metrics"]["gain_loss_ratio"], ""),
        ("최장 연속 상승", result["metrics"]["positive_streak"], "일"),
        ("최장 연속 하락", result["metrics"]["negative_streak"], "일"),
        ("베타", result["metrics"].get("beta", 0), ""),
        ("알파", result["metrics"].get("alpha", 0), "%"),
        ("정보 비율", result["metrics"].get("information_ratio", 0), ""),
        ("칼마 비율", result["metrics"].get("calmar_ratio", 0), ""),
    ]
    
    # 벤치마크 지표 키 매핑 정의
    metric_key_mapping = {
        "총 수익률": "total_return",
        "연간 수익률": "annualized_return",
        "샤프 비율": "sharpe_ratio",
        "소르티노 비율": "sortino_ratio",
        "최대 낙폭": "max_drawdown",
        "변동성": "annualized_volatility",
        "승률": "win_rate",
        "손익비": "gain_loss_ratio",
        "최장 연속 상승": "positive_streak",
        "최장 연속 하락": "negative_streak",
        "베타": "beta",
        "알파": "alpha",
        "정보 비율": "information_ratio",
        "칼마 비율": "calmar_ratio",
    }
    
    # 각 지표별 행 추가
    for name, value, unit in metrics_info:
        if "%" in unit:
            row = [name, f"{value*100:.2f}{unit}"]
        elif value > 100:
            row = [name, f"{value:.2f}{unit}"]
        else:
            row = [name, f"{value:.4f}{unit}"]
        
        # 벤치마크 지표 추가 (있는 경우)
        if benchmark_results:
            metric_key = metric_key_mapping.get(name, name.lower().replace(" ", "_").replace("/", "_"))
            
            for ticker, data in benchmark_results.items():
                benchmark_value = data["metrics"].get(metric_key, 0)
                
                if "%" in unit:
                    row.append(f"{benchmark_value*100:.2f}{unit}")
                elif benchmark_value > 100:
                    row.append(f"{benchmark_value:.2f}{unit}")
                else:
                    row.append(f"{benchmark_value:.4f}{unit}")
        
        rows.append(row)
    
    # 테이블 생성
    table = tabulate(rows, headers, tablefmt="grid")
    full_table = f"{title}\n\n{table}"
    
    # 파일로 저장
    if save_path:
        # 저장 경로의 디렉토리가 없으면 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        with open(save_path, "w") as f:
            f.write(full_table)
    
    return full_table

def plot_backtest_results(result, benchmark_results=None, save_path=None, title="Backtest Results"):
    """
    백테스팅 결과를 시각화합니다.
    
    Args:
        result: 백테스팅 결과
        benchmark_results: 벤치마크 결과 (선택적)
        save_path: 저장 경로
        title: 그래프 제목
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 날짜 변환
    dates = pd.to_datetime(result["test_dates"])
    
    # 1. 포트폴리오 가치 곡선
    portfolio_values = result["portfolio_values"]
    ax1.plot(dates, portfolio_values, 'b-', label='Portfolio', linewidth=2)
    
    # 벤치마크 추가 (있는 경우)
    if benchmark_results:
        colors = ['red', 'green', 'purple', 'orange']
        for i, (ticker, data) in enumerate(benchmark_results.items()):
            benchmark_values = data["values"]
            if len(benchmark_values) == len(dates):
                ax1.plot(dates, benchmark_values, color=colors[i % len(colors)], 
                         linestyle='-', label=f'{ticker}', linewidth=1.5, alpha=0.8)
    
    # 첫 번째 그래프 스타일링
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', labelsize=10)
    
    # x축 날짜 포맷팅
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    
    # 2. 일별 수익률 그래프
    daily_returns = result["daily_returns"]
    ax2.bar(dates, daily_returns * 100, color='blue', alpha=0.6, width=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 두 번째 그래프 스타일링
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', labelsize=10)
    
    # x축 날짜 포맷팅
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    if save_path:
        # 저장 경로의 디렉토리가 없으면 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_asset_allocation(weights, dates, asset_names, save_path=None, title="Asset Allocation Over Time"):
    """
    시간에 따른 자산 배분 변화를 시각화합니다.
    
    Args:
        weights: 자산 가중치 배열 (시간 x 자산)
        dates: 날짜 배열
        asset_names: 자산 이름 목록
        save_path: 저장 경로
        title: 그래프 제목
    """
    plt.figure(figsize=(14, 8))
    
    # 날짜 변환
    dates = pd.to_datetime(dates)
    
    # 누적 차트 데이터 준비
    df = pd.DataFrame(weights, index=dates)
    
    # 자산 이름 확인 및 조정
    n_assets = weights.shape[1] - 1  # 마지막 열은 현금
    
    if len(asset_names) < n_assets:
        # 자산 이름이 부족한 경우 채움
        asset_names = list(asset_names) + [f"Asset_{i+1}" for i in range(len(asset_names), n_assets)]
    
    # 열 이름 설정 (마지막은 현금)
    column_names = asset_names[:n_assets] + ["Cash"]
    df.columns = column_names
    
    # 면적 그래프 그리기
    ax = df.plot.area(stacked=True, alpha=0.7, figsize=(14, 8))
    
    # 그래프 스타일링
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=min(6, len(column_names)), fontsize=10)
    
    # x축 날짜 포맷팅
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    
    # y축 범위 설정
    plt.ylim(0, 1)
    
    # 저장
    if save_path:
        # 저장 경로의 디렉토리가 없으면 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_agent_agreement(individual_actions, dates, save_path=None):
    """
    앙상블 에이전트 간 행동 일치도를 시각화합니다.
    
    Args:
        individual_actions: 각 에이전트의 행동 리스트
        dates: 날짜 배열
        save_path: 저장 경로
    """
    plt.figure(figsize=(14, 8))
    
    # 날짜 변환 (관측 날짜보다 하루 적음)
    if len(dates) > len(individual_actions) + 1:
        plot_dates = pd.to_datetime(dates[1:len(individual_actions)+1])
    else:
        plot_dates = pd.to_datetime(dates[1:])
    
    # 일치도 계산 (표준편차의 역수로 측정)
    agreement_scores = []
    for day_actions in individual_actions:
        day_actions_array = np.array(day_actions)
        # 각 자산별 표준편차 계산
        std_per_asset = np.std(day_actions_array, axis=0)
        # 평균 표준편차 (0에 가까울수록 일치도 높음)
        mean_std = np.mean(std_per_asset)
        # 일치도 점수 (0~1, 높을수록 일치)
        agreement = 1.0 / (1.0 + 10.0 * mean_std)
        agreement_scores.append(agreement)
    
    # 일치도 그래프 그리기
    plt.plot(plot_dates, agreement_scores, 'g-', linewidth=2)
    
    # 그래프 스타일링
    plt.title('Ensemble Agent Agreement', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Agreement Score (higher=better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # y축 범위 설정
    plt.ylim(0, 1)
    
    # x축 날짜 포맷팅
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    
    # 저장
    if save_path:
        # 저장 경로의 디렉토리가 없으면 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 