"""
상수 정의 모듈

이 모듈은 FinFlow RL 프로젝트에서 사용되는 모든 상수 값을 정의합니다.
학습 설정, 환경 파라미터, 경로 설정 등 프로젝트 전반에 사용되는 상수들을 중앙 집중화하여 관리합니다.
"""

import torch
import os

# --- 상수 정의 ---
# 학습 및 앙상블 설정
NUM_EPISODES = 200  # 학습 에피소드 수 (100→200으로 증가)
ENSEMBLE_SIZE = 10 # 앙상블 에이전트 수 (8→10으로 증가)

# 학습 스케줄 및 Early Stopping 설정
EARLY_STOPPING_PATIENCE = 30  # 성능 향상이 없는 최대 에피소드 수 (100→30으로 감소)
LR_SCHEDULER_T_MAX = 300  # Cosine Annealing 주기 (500→300으로 감소)
LR_SCHEDULER_ETA_MIN = 5e-6  # 최소 학습률 (1e-6→5e-6으로 증가)
VALIDATION_INTERVAL = 5  # 검증 수행 간격 (에피소드) (10→5로 감소)
VALIDATION_EPISODES = 8  # 검증 시 평가할 에피소드 수 (5→8로 증가)

# 벤치마크 설정
BENCHMARK_TICKERS = ["SPY", "QQQ"]  # S&P 500 ETF, Nasdaq 100 ETF
USE_BENCHMARK = True  # 벤치마크 비교 기능 사용 여부

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 분석 대상 주식 티커 목록
STOCK_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "AMD",
    "TSLA",
    "JPM",
    "JNJ",
    "PG",
    "V",
]

# 학습/테스트 데이터 기간 설정
TRAIN_START_DATE = "2008-01-02"
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2024-12-31"

# main.py에서 사용하는 변수명과 동기화
START_DATE = TRAIN_START_DATE
END_DATE = TEST_END_DATE
TRAIN_TEST_SPLIT_RATIO = 0.8  # 학습/테스트 데이터 분할 비율

# 포트폴리오 초기 설정
INITIAL_CASH = 1e6
COMMISSION_RATE = 0.0003  # 수수료 현실화 (0.0005→0.0003)
# 행동 변화 페널티 계수
ACTION_PENALTY_COEF = 0.0003  # (0.0005→0.0003로 감소)

# 행동 스케일링 계수 조정
DIRICHLET_SCALE_FACTOR = 3.0  # (5.0→3.0으로 감소)

# 온도 스케일링 파라미터 조정
SOFTMAX_TEMPERATURE_INITIAL = 1.5  # (1.0→1.5로 증가)
SOFTMAX_TEMPERATURE_MIN = 0.3  # (0.2→0.3으로 증가)
SOFTMAX_TEMPERATURE_DECAY = 0.99  # (0.995→0.99로 조정)

# 보상 누적 기간 (K-일)
REWARD_ACCUMULATION_DAYS = 3  # (5→3으로 감소, 더 빠른 피드백)

# 보상 함수 관련 설정
REWARD_SHARPE_WINDOW = 12  # Sharpe ratio 계산 윈도우 (15→12로 감소)
REWARD_RETURN_WEIGHT = 0.8  # 수익률 가중치 (0.7→0.8로 증가)
REWARD_SHARPE_WEIGHT = 0.2  # Sharpe ratio 가중치 (0.3→0.2로 감소)
REWARD_DRAWDOWN_PENALTY = 0.1  # 드로우다운 페널티 계수 (0.15→0.1로 감소) 
REWARD_VOL_SCALE_MIN = 0.92  # 변동성 기반 클리핑 최소값 (0.9→0.92)
REWARD_VOL_SCALE_MAX = 1.08  # 변동성 기반 클리핑 최대값 (1.1→1.08)
REWARD_LONG_TERM_BONUS = 0.08  # 장기 보상 보너스 계수 (0.05→0.08)
REWARD_NEGATIVE_WEIGHT = 1.0  # 음수 보상 가중치 (1.1→1.0)
# 보상 클리핑 범위 수정
REWARD_CLIP_MIN = -1.0  # 보상 클리핑 최소값 (-1.5→-1.0)
REWARD_CLIP_MAX = 1.0  # 보상 클리핑 최대값 (1.5→1.0)
# Sharpe ratio 클리핑 값 수정
SHARPE_RATIO_CLIP = 1.5  # (2.0→1.5)

# PPO 하이퍼파라미터 수정
DEFAULT_HIDDEN_DIM = 128  # 모델 크기 유지
DEFAULT_LR = 1e-4  # 학습률 (3e-5→1e-4)
DEFAULT_GAMMA = 0.98  # 할인율 (0.99→0.98)
DEFAULT_K_EPOCHS = 6  # PPO 에폭 수 (8→6)
DEFAULT_EPS_CLIP = 0.1  # PPO 클리핑 파라미터 (0.15→0.1)
PPO_UPDATE_TIMESTEP = 200  # PPO 업데이트 주기 (250→200)
BATCH_SIZE = 256  # 배치 사이즈 (128→256)
GRADIENT_CLIP = 0.3  # 그래디언트 클리핑 값 (0.5→0.3)
ENTROPY_COEF = 0.02  # 엔트로피 보너스 계수 (0.01→0.02)
DEFAULT_ENTROPY_COEF = 0.02  # PPO 클래스에서 사용할 엔트로피 계수 (0.01→0.02)
CRITIC_COEF = 0.4  # 크리틱 계수 (0.5→0.4)

# 환경 설정
MAX_EPISODE_LENGTH = 504  # 환경의 최대 에피소드 길이

# 상태/보상 정규화 설정 - 핵심 수정
NORMALIZE_STATES = True
CLIP_OBS = 3.0  # (5.0→3.0)
CLIP_REWARD = 0.8  # 보상 클리핑 범위 (1.0→0.8)
RMS_EPSILON = 1e-4  # (1e-5→1e-4)

# GAE 설정
LAMBDA_GAE = 0.9  # (0.92→0.9)

# 모델, 데이터 캐시, 결과 저장 경로
MODEL_SAVE_PATH = "models"
DATA_SAVE_PATH = "data"
RESULTS_BASE_PATH = "results"  # 새로운 결과 저장 기본 경로

# 설명 가능한 AI (XAI) 관련 설정
INTEGRATED_GRADIENTS_STEPS = 50
# XAI_SAMPLE_COUNT는 통합 그래디언트, SHAP 등 계산 비용이 높은 
# XAI 방법을 위한 기본 샘플 수입니다. 계산 시간과 정확도 간 균형을 맞춥니다.
XAI_SAMPLE_COUNT = 5

# 피처 이름 정의 (데이터 처리 순서와 일치)
FEATURE_NAMES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MACD",
    "RSI",
    "MA14",
    "MA21",
    "MA100",
] 