#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FinFlow RL: 강화학습 기반 포트폴리오 최적화

이 프로그램은 강화학습을 사용하여 주식 포트폴리오 관리 전략을 학습합니다.
다양한 자산에 대한 포트폴리오 배분 의사결정을 자동화하는 데 중점을 둡니다.
PPO(Proximal Policy Optimization) 알고리즘 기반으로 구현되었습니다.
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import traceback
import gc

from src.constants import (
    STOCK_TICKERS,
    START_DATE, 
    END_DATE, 
    TRAIN_TEST_SPLIT_RATIO,
    BENCHMARK_TICKERS, 
    USE_BENCHMARK,
    DEVICE,
    RESULTS_BASE_PATH,
    ENSEMBLE_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LR,
    DEFAULT_GAMMA,
    DEFAULT_EPS_CLIP,
    DEFAULT_K_EPOCHS,
    NORMALIZE_STATES
)

from src.utils.data_utils import (
    fetch_and_preprocess_data, 
    fetch_benchmark_data
)
from src.utils.logging_utils import setup_logger
from src.environment.portfolio_env import StockPortfolioEnv
from src.models.ppo import PPO
from src.train.train import (
    train_agent, 
    train_ensemble, 
    load_ensemble
)
from src.evaluation.evaluation import (
    backtest, 
    ensemble_backtest
)
from src.xai.explainable_ai import run_model_interpretability


def log_section_header(logger, message):
    """로그에 구분선과 함께 섹션 헤더를 출력합니다."""
    logger.info("="*70)
    logger.info(f">>> {message}")
    logger.info("="*70)


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="FinFlow RL: 강화학습 기반 포트폴리오 최적화")
    
    # 실행 모드 인자
    parser.add_argument("mode", type=str, choices=['all', 'train', 'ensemble_train', 'backtest', 'ensemble_backtest', 'xai'],
                        help="실행 모드 (all: 전체 실행, train: 단일 모델 학습, ensemble_train: 앙상블 모델 학습, "
                        "backtest: 단일 모델 백테스팅, ensemble_backtest: 앙상블 모델 백테스팅, xai: 설명 가능한 AI 분석)")
    
    # 데이터 관련 인자
    parser.add_argument("--tickers", type=str, nargs='+', default=STOCK_TICKERS,
                        help="분석할 주식 티커 목록 (기본값: constants.py에 정의된 STOCK_TICKERS)")
    parser.add_argument("--start_date", type=str, default=START_DATE,
                        help="데이터 시작 날짜 (YYYY-MM-DD 형식)")
    parser.add_argument("--end_date", type=str, default=END_DATE,
                        help="데이터 종료 날짜 (YYYY-MM-DD 형식)")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_TEST_SPLIT_RATIO,
                        help="훈련 데이터 비율 (0.0 ~ 1.0)")
    
    # 모델 관련 인자
    parser.add_argument("--ensemble_size", type=int, default=ENSEMBLE_SIZE,
                        help="앙상블 모델 수")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM,
                        help="신경망 은닉층 차원")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="학습률")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="할인율")
    parser.add_argument("--k_epochs", type=int, default=DEFAULT_K_EPOCHS,
                        help="PPO 업데이트 에폭 수")
    parser.add_argument("--eps_clip", type=float, default=DEFAULT_EPS_CLIP,
                        help="PPO 클리핑 파라미터")
    
    # 학습 및 평가 관련 인자
    parser.add_argument("--model_path", type=str, default=None,
                        help="모델 저장/로드 경로 (기본값: results 내 자동 생성)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="결과 저장 디렉토리 (기본값: results 내 자동 생성)")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="EMA(Exponential Moving Average) 모델 사용 여부")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false",
                        help="EMA(Exponential Moving Average) 모델 사용하지 않음")
    parser.add_argument("--use_benchmark", action="store_true", default=USE_BENCHMARK,
                        help="벤치마크 비교 사용 여부")
    parser.add_argument("--no_benchmark", dest="use_benchmark", action="store_false",
                        help="벤치마크 비교 사용하지 않음")
    parser.add_argument("--benchmark_tickers", type=str, nargs='+', default=BENCHMARK_TICKERS,
                        help="벤치마크 티커 목록 (기본값: constants.py에 정의된 BENCHMARK_TICKERS)")
    
    # XAI 관련 인자
    parser.add_argument("--xai_methods", type=str, nargs='+', default=['shap', 'sensitivity', 'decision', 'integrated_gradients'],
                        help="XAI 분석 방법 (여러 개 지정 가능: shap, sensitivity, decision, integrated_gradients)")
    parser.add_argument("--xai_samples", type=int, default=50,
                        help="XAI 분석에 사용할 샘플 수")
    
    return parser.parse_args()


def prepare_data(args, logger):
    """데이터를 준비하고 학습/테스트 분할을 수행합니다."""
    
    log_section_header(logger, "데이터 준비 시작")
    logger.info(f"데이터 기간: {args.start_date} ~ {args.end_date}")
    logger.info(f"대상 티커: {args.tickers}")
    
    # 데이터 로드 및 전처리
    data_array, dates = fetch_and_preprocess_data(
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers
    )
    
    if data_array is None or dates is None:
        logger.error("데이터 로드 실패. 프로그램을 종료합니다.")
        sys.exit(1)
    
    logger.info(f"로드된 데이터 형태: {data_array.shape}, 날짜 수: {len(dates)}")
    
    # 학습/테스트 데이터 분할
    split_idx = int(len(data_array) * args.train_ratio)
    train_data = data_array[:split_idx]
    test_data = data_array[split_idx:]
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    logger.info(f"학습 데이터: {train_data.shape}, 학습 기간: {train_dates[0]} ~ {train_dates[-1]}")
    logger.info(f"테스트 데이터: {test_data.shape}, 테스트 기간: {test_dates[0]} ~ {test_dates[-1]}")
    
    # 벤치마크 데이터 로드 (필요한 경우)
    benchmark_data = None
    if args.use_benchmark:
        logger.info(f"벤치마크 데이터 로드: {args.benchmark_tickers}")
        try:
            # 벤치마크 데이터 로드
            benchmark_data = fetch_benchmark_data(
                benchmark_tickers=args.benchmark_tickers,
                start_date=test_dates[0].strftime("%Y-%m-%d"),
                end_date=test_dates[-1].strftime("%Y-%m-%d")
            )
            
            # 벤치마크 데이터 로드 결과 확인
            if benchmark_data and len(benchmark_data) > 0:
                logger.info(f"벤치마크 데이터 로드 완료: {list(benchmark_data.keys())} ({len(benchmark_data)} 종목)")
            else:
                logger.warning("벤치마크 데이터 로드 실패 또는 비어있음. 벤치마크 없이 진행합니다.")
                args.use_benchmark = False
        except Exception as e:
            logger.warning(f"벤치마크 데이터 로드 중 오류 발생: {e}. 벤치마크 없이 진행합니다.")
            args.use_benchmark = False
    
    return train_data, test_data, train_dates, test_dates, benchmark_data


def create_agent(args, train_data, logger):
    """단일 PPO 에이전트를 생성합니다."""
    
    # 에이전트 생성
    n_assets = train_data.shape[1]
    n_features = train_data.shape[2]
    
    # model_path가 None이면 기본 경로 설정
    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(args.results_dir, "models")
        logger.info(f"모델 경로가 지정되지 않아 기본 경로로 설정됨: {model_path}")
    
    agent = PPO(
        n_assets=n_assets,
        n_features=n_features,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        k_epochs=args.k_epochs,
        eps_clip=args.eps_clip,
        model_path=model_path,
        logger=logger,
        use_ema=args.use_ema
    )
    
    logger.info(f"PPO 에이전트 생성 완료: {n_assets} 자산, {n_features} 피처, 은닉층 차원: {args.hidden_dim}")
    logger.info(f"학습 파라미터: lr={args.lr}, gamma={args.gamma}, k_epochs={args.k_epochs}, eps_clip={args.eps_clip}")
    
    return agent


def load_agent(args, train_data, logger):
    """저장된 모델을 로드하여 에이전트를 생성합니다."""
    
    # model_path 확인 및 설정
    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(args.results_dir, "models")
        logger.warning(f"모델 경로가 지정되지 않아 기본 경로로 설정됨: {model_path}")
    
    # 에이전트 생성
    n_assets = train_data.shape[1]
    n_features = train_data.shape[2]
    
    agent = PPO(
        n_assets=n_assets,
        n_features=n_features,
        hidden_dim=args.hidden_dim,
        model_path=model_path,
        logger=logger,
        use_ema=args.use_ema
    )
    
    # 모델 로드
    model_file = os.path.join(model_path, "best_model.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(model_path, "final_model.pth")
    
    if not os.path.exists(model_file):
        logger.error(f"모델 파일을 찾을 수 없습니다: {model_file}")
        sys.exit(1)
    
    success = agent.load_model(model_file)
    if not success:
        logger.error(f"모델 로드 실패: {model_file}")
        sys.exit(1)
    
    logger.info(f"모델 로드 성공: {model_file}")
    
    return agent


def train_mode(args, train_data, test_data, train_dates, test_dates, logger):
    """단일 모델 학습 모드를 실행합니다."""
    
    # 결과 디렉토리 생성
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join(RESULTS_BASE_PATH, f"finflow_all_{timestamp}")
        os.makedirs(args.results_dir, exist_ok=True)
    
        # 명시적으로 체크포인트 디렉토리 생성
        checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 학습 과정 로깅을 위한 별도 디렉토리
        logs_dir = os.path.join(args.results_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # 로깅 파일 핸들러 추가
        log_file = os.path.join(args.results_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    log_section_header(logger, "단일 모델 학습 시작")
    logger.info(f"학습 결과 저장 디렉토리: {args.results_dir}")
    
    # 에이전트 생성
    agent = create_agent(args, train_data, logger)
    
    # 모델 저장 경로 설정
    agent.model_path = args.results_dir
    
    # 환경 생성
    train_env = StockPortfolioEnv(train_data)
    val_env = StockPortfolioEnv(test_data)
    
    # 학습 실행
    logger.info("모델 학습 시작")
    start_time = time.time()
    
    results = train_agent(
        env=train_env,
        ppo_agent=agent,
        validate_env=val_env,
        run_dir=args.results_dir,
        logger=logger
    )
    
    training_time = time.time() - start_time
    logger.info(f"모델 학습 완료. 소요 시간: {training_time:.2f}초")
    
    # 학습 결과 반환
    return agent, results


def ensemble_train_mode(args, train_data, test_data, train_dates, test_dates, logger):
    """앙상블 모델 학습 모드를 실행합니다."""
    
    # 결과 디렉토리 생성
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join(RESULTS_BASE_PATH, f"finflow_ensemble_{timestamp}")
        os.makedirs(args.results_dir, exist_ok=True)
    
    # 앙상블 모델 디렉토리 생성
    ensemble_dir = os.path.join(args.results_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    log_section_header(logger, "앙상블 모델 학습 시작")
    logger.info(f"앙상블 학습 결과 저장 디렉토리: {args.results_dir}")
    logger.info(f"앙상블 모델 수: {args.ensemble_size}")
    
    # 앙상블 학습 실행
    start_time = time.time()
    
    ensemble_agents = train_ensemble(
        train_data=train_data,
        test_data=test_data,
        train_dates=train_dates,
        test_dates=test_dates,
        ensemble_size=args.ensemble_size,
        ensemble_dir=ensemble_dir,
        logger=logger
    )
    
    training_time = time.time() - start_time
    logger.info(f"앙상블 모델 학습 완료. 모델 수: {len(ensemble_agents)}, 소요 시간: {training_time:.2f}초")
    
    # 앙상블 결과 반환
    return ensemble_agents


def backtest_mode(args, train_data, test_data, train_dates, test_dates, benchmark_data, logger):
    """단일 모델 백테스팅 모드를 실행합니다."""
    
    # 에이전트 로드
    agent = load_agent(args, train_data, logger)
    
    # 결과 디렉토리 생성
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join(RESULTS_BASE_PATH, f"finflow_backtest_{timestamp}")
        os.makedirs(args.results_dir, exist_ok=True)
    
    backtest_dir = os.path.join(args.results_dir, "backtest")
    os.makedirs(backtest_dir, exist_ok=True)
    
    log_section_header(logger, "단일 모델 백테스팅 시작")
    logger.info(f"백테스팅 결과 저장 디렉토리: {backtest_dir}")
    logger.info(f"백테스팅 기간: {test_dates[0]} ~ {test_dates[-1]}")
    
    # 백테스팅 실행
    start_time = time.time()
    
    results = backtest(
        ppo_agent=agent,
        test_data=test_data,
        test_dates=test_dates,
        benchmark_data=benchmark_data,
        log_dir=backtest_dir,
        logger=logger,
        use_ema=args.use_ema
    )
    
    backtest_time = time.time() - start_time
    logger.info(f"백테스팅 완료. 소요 시간: {backtest_time:.2f}초")
    
    # 결과 요약 출력
    if "metrics" in results:
        metrics = results["metrics"]
        logger.info("===== 백테스팅 결과 요약 =====")
        logger.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
        logger.info(f"연간 수익률: {metrics['annualized_return']*100:.2f}%")
        logger.info(f"연간 변동성: {metrics['annualized_volatility']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"최대 낙폭: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"승률: {metrics['win_rate']*100:.2f}%")
    
    return results


def ensemble_backtest_mode(args, train_data, test_data, train_dates, test_dates, benchmark_data, logger):
    """앙상블 모델 백테스팅 모드를 실행합니다."""
    
    # 앙상블 에이전트 로드
    if args.model_path is None:
        logger.error("앙상블 모델 경로가 지정되지 않았습니다. --model_path 인자를 사용하여 모델 경로를 지정하세요.")
        sys.exit(1)
    
    log_section_header(logger, "앙상블 모델 백테스팅 시작")
    logger.info(f"앙상블 모델 로드: {args.model_path}")
    
    ensemble_agents = load_ensemble(args.model_path, logger)
    
    if not ensemble_agents:
        logger.error("앙상블 모델 로드 실패")
        sys.exit(1)
    
    logger.info(f"앙상블 모델 로드 완료: {len(ensemble_agents)} 에이전트")
    
    # 결과 디렉토리 생성
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join(RESULTS_BASE_PATH, f"finflow_ensemble_backtest_{timestamp}")
        os.makedirs(args.results_dir, exist_ok=True)
    
    ensemble_backtest_dir = os.path.join(args.results_dir, "ensemble_backtest")
    os.makedirs(ensemble_backtest_dir, exist_ok=True)
    
    logger.info(f"앙상블 백테스팅 결과 저장 디렉토리: {ensemble_backtest_dir}")
    logger.info(f"백테스팅 기간: {test_dates[0]} ~ {test_dates[-1]}")
    
    # 앙상블 백테스팅 실행
    start_time = time.time()
    
    results = ensemble_backtest(
        ensemble_agents=ensemble_agents,
        test_data=test_data,
        test_dates=test_dates,
        benchmark_data=benchmark_data,
        log_dir=ensemble_backtest_dir,
        logger=logger,
        use_ema=args.use_ema,
        voting_method='mean'  # 'mean', 'median', 'rank' 중 선택
    )
    
    backtest_time = time.time() - start_time
    logger.info(f"앙상블 백테스팅 완료. 소요 시간: {backtest_time:.2f}초")
    
    # 결과 요약 출력
    if "metrics" in results:
        metrics = results["metrics"]
        logger.info("===== 앙상블 백테스팅 결과 요약 =====")
        logger.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
        logger.info(f"연간 수익률: {metrics['annualized_return']*100:.2f}%")
        logger.info(f"연간 변동성: {metrics['annualized_volatility']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"최대 낙폭: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"승률: {metrics['win_rate']*100:.2f}%")
    
    return results


def xai_mode(args, train_data, test_data, train_dates, test_dates, logger):
    """설명 가능한 AI(XAI) 분석 모드를 실행합니다."""
    
    # 에이전트 로드
    agent = load_agent(args, train_data, logger)
    
    # 결과 디렉토리 생성
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join(RESULTS_BASE_PATH, f"finflow_xai_{timestamp}")
        os.makedirs(args.results_dir, exist_ok=True)
    
    xai_dir = os.path.join(args.results_dir, "xai")
    os.makedirs(xai_dir, exist_ok=True)
    
    log_section_header(logger, "XAI 분석 시작")
    logger.info(f"XAI 분석 결과 저장 디렉토리: {xai_dir}")
    
    # integrated_gradients를 methods에 추가
    if 'integrated_gradients' not in args.xai_methods:
        args.xai_methods.append('integrated_gradients')
    
    # XAI 분석 실행
    logger.info(f"XAI 분석 방법: {args.xai_methods}, 샘플 수: {args.xai_samples}")
    start_time = time.time()
    
    results = run_model_interpretability(
        agent=agent,
        test_data=test_data,
        test_dates=test_dates,
        methods=args.xai_methods,
        save_dir=xai_dir,
        logger=logger,
        use_ema=args.use_ema
    )
    
    xai_time = time.time() - start_time
    logger.info(f"XAI 분석 완료. 소요 시간: {xai_time:.2f}초")
    
    # 결과 요약 출력
    if "combined_importance" in results:
        combined_importance = results["combined_importance"]
        feature_names = results.get("feature_names", [f"특성 {i}" for i in range(len(combined_importance))])
        
        # 정렬된 중요도 출력
        sorted_indices = np.argsort(combined_importance)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importance = combined_importance[sorted_indices]
        
        logger.info("===== XAI 분석 결과 요약 =====")
        logger.info("특성 중요도 (상위 5개):")
        for i in range(min(5, len(sorted_features))):
            logger.info(f"{sorted_features[i]}: {sorted_importance[i]*100:.2f}%")
    
    return results


def main():
    """메인 함수"""
    
    # 인자 파싱
    args = parse_args()
    
    # 결과 디렉토리 생성 (체계적인 구조)
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join(RESULTS_BASE_PATH, f"finflow_{args.mode}_{timestamp}")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 하위 디렉토리 미리 생성
    if args.mode == 'all':
        os.makedirs(os.path.join(args.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, "ensemble"), exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, "backtest"), exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, "ensemble_backtest"), exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, "xai"), exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(args.results_dir)
    
    # 시작 메시지
    log_section_header(logger, f"FinFlow RL 시작: 모드={args.mode}, 시간={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # GPU 정보 출력
        if torch.cuda.is_available():
            logger.info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA 버전: {torch.version.cuda}")
        else:
            logger.info("GPU 사용 불가능, CPU 모드로 실행")
        
        # 데이터 준비
        train_data, test_data, train_dates, test_dates, benchmark_data = prepare_data(args, logger)
        
        # 모드에 따라 실행
        if args.mode == 'train':
            agent, results = train_mode(args, train_data, test_data, train_dates, test_dates, logger)
        
        elif args.mode == 'ensemble_train':
            ensemble_agents = ensemble_train_mode(args, train_data, test_data, train_dates, test_dates, logger)
        
        elif args.mode == 'backtest':
            results = backtest_mode(args, train_data, test_data, train_dates, test_dates, benchmark_data, logger)
        
        elif args.mode == 'ensemble_backtest':
            results = ensemble_backtest_mode(args, train_data, test_data, train_dates, test_dates, benchmark_data, logger)
        
        elif args.mode == 'xai':
            results = xai_mode(args, train_data, test_data, train_dates, test_dates, logger)
        
        elif args.mode == 'all':
            # 학습 후 백테스팅 및 XAI 분석까지 모두 수행
            # 경로 정의
            models_dir = os.path.join(args.results_dir, "models")
            ensemble_dir = os.path.join(args.results_dir, "ensemble")
            backtest_dir = os.path.join(args.results_dir, "backtest")
            ensemble_backtest_dir = os.path.join(args.results_dir, "ensemble_backtest")
            xai_dir = os.path.join(args.results_dir, "xai")
            
            # 각 단계별 실행
            log_section_header(logger, "단일 모델 학습 시작")
            args.model_path = models_dir
            agent, train_results = train_mode(args, train_data, test_data, train_dates, test_dates, logger)
            
            log_section_header(logger, "앙상블 모델 학습 시작")
            ensemble_agents = train_ensemble(
                train_data=train_data,
                test_data=test_data,
                train_dates=train_dates,
                test_dates=test_dates,
                ensemble_size=args.ensemble_size,
                ensemble_dir=ensemble_dir,
                logger=logger
            )
            
            log_section_header(logger, "단일 모델 백테스팅 시작")
            backtest_results = backtest(
                ppo_agent=agent,
                test_data=test_data,
                test_dates=test_dates,
                benchmark_data=benchmark_data,
                log_dir=backtest_dir,
                logger=logger,
                use_ema=args.use_ema
            )
            
            log_section_header(logger, "앙상블 모델 백테스팅 시작")
            ensemble_backtest_results = ensemble_backtest(
                ensemble_agents=ensemble_agents,
                test_data=test_data,
                test_dates=test_dates,
                benchmark_data=benchmark_data,
                log_dir=ensemble_backtest_dir,
                logger=logger,
                use_ema=args.use_ema,
                voting_method='mean'
            )
            
            log_section_header(logger, "XAI 분석 시작")
            # integrated_gradients를 methods에 추가
            if 'integrated_gradients' not in args.xai_methods:
                args.xai_methods.append('integrated_gradients')
                
            xai_results = run_model_interpretability(
                agent=agent,
                test_data=test_data,
                test_dates=test_dates,
                methods=args.xai_methods,
                save_dir=xai_dir,
                logger=logger,
                use_ema=args.use_ema
            )
            
            results = {
                "train_results": train_results,
                "ensemble_train_results": {"num_agents": len(ensemble_agents)},
                "backtest_results": backtest_results,
                "ensemble_backtest_results": ensemble_backtest_results,
                "xai_results": xai_results
            }
        
        # 처리 완료 메시지
        log_section_header(logger, f"FinFlow RL 완료: 모드={args.mode}, 시간={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"결과 저장 위치: {args.results_dir}")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
