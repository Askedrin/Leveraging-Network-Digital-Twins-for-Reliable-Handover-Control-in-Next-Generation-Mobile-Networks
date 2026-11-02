"""
config.py
=========
ONE shared configuration for all six architectures, so the Table-III comparison
is fair (identical data, split, sequence length, optimizer, schedule, epochs).
Change values here, not inside train.py.

The normalization convention is deliberately pinned to the repo's inference-time
transform so that models trained here are *directly consumable* by the decision
agent without touching its denormalization:

    rsrp_norm = (rsrp_dBm + 120) / 60          # -> roughly [0, 1] over [-120,-60]
    rsrp_dBm  = rsrp_norm * 60 - 120           # the agent's exact denorm

Because RSRP is normalized by this FIXED affine map (not a data-fitted scaler),
MAE/RMSE computed on de-normalized predictions are already in TRUE dB. The other
11 features are scaled by a data-fitted MinMaxScaler saved alongside (see
data_prep.py). Together they are persisted as a single artifact `scaler.joblib`.
"""

import os

# --- paths ---------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR   = os.path.join(HERE, "data", "raw")      # put UCC CSVs here
PROC_DATA_DIR  = os.path.join(HERE, "data", "processed")
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
RESULTS_DIR    = os.path.join(HERE, "results")
SCALER_PATH    = os.path.join(PROC_DATA_DIR, "scaler.joblib")

# --- predictor contract (the canonical inference anchor; do NOT change) ---- #
# The 12 engineered features, in the exact order the repo feeds them
# (best2.txt line 1081; identical in lll.ipynb / code1.ipynb).
LSTM_FEATURES = [
    "Longitude", "Latitude", "Speed", "RSRP", "Distance_serving",
    "Heading", "Hour_sin", "Hour_cos", "Handover_recent",
    "RSRP_trend", "Speed_normalized", "Distance_normalized",
]
TARGET_FEATURE = "RSRP"
INPUT_SIZE     = len(LSTM_FEATURES)     # 12
OUTPUT_SIZE    = 1                       # single-step (see PREDICTION_HORIZON note)
SEQUENCE_LENGTH = 15                     # H = 15

# RSRP fixed-range normalization (matches inference denorm r = 60*r_norm - 120)
RSRP_MIN_DBM = -120.0
RSRP_MAX_DBM =  -60.0

# Prediction horizon. The repo inference is 1-step; the paper claims k=5. This
# pipeline trains a 1-step model (OUTPUT_SIZE=1) and produces the k-step forecast
# by AUTOREGRESSIVE rollout at evaluation time (evaluate.py / train.py), which is
# the honest reconciliation of the "1-step code vs k=5 paper" conflict flagged in
# reconciliation.md. Set K_ROLLOUT to the horizon you want reported.
K_ROLLOUT = 5

# --- split (by trace, NOT by row — prevents leakage) ---------------------- #
# UCC/Raca traces are per-drive CSV files. We split on the FILE/trace id.
SPLIT_BY = "trace"           # 'trace' (required by spec) — row-splitting is a bug
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SPLIT_SEED = 42

# --- shared training config (identical for all six) ----------------------- #
EPOCHS          = 100
BATCH_SIZE      = 64
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5
GRAD_CLIP_NORM  = 1.0
EARLY_STOP_PATIENCE = 20      # epochs w/o val improvement
LR_PATIENCE     = 10          # ReduceLROnPlateau patience
LR_FACTOR       = 0.5
LOSS            = "mse"       # reported metrics (MAE/RMSE/R2) are loss-independent
NUM_WORKERS     = 4
SEED            = 42

# Which architectures to train (all six -> Table III).
ARCHITECTURES_TO_TRAIN = [
    "ImprovedLSTM", "BiLSTM", "LSTMAttention", "TCN", "Transformer", "CNNLSTM",
]
