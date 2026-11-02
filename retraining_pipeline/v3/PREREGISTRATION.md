# Phase 1a — forecaster on real traces (pre-registration)

Written 2026-10-27 before any v3 forecaster training. Results reported whatever they are.

## Why a v3 pipeline
The earlier pipeline (`../data_prep.py`) (i) mixed radio technologies — in HSPA+/UMTS/EDGE/GPRS rows
(14 % of samples) the `RSRP` field is not LTE/NR RSRP (saturates at −51 dBm) and `NRxRSRP` is a
constant −51 placeholder; (ii) kept invalid LTE values (down to −200 dBm); (iii) clipped all RSRP
to [−120, −60] dBm, turning both into long constant runs; (iv) trained one-step models only.

## Data
- Same 60 UCC driving traces (operator B), same by-trace split (seed 42, 70/15/15 → 42/9/9).
- Keep rows with NetworkMode ∈ {LTE, 5G} and RSRP ∈ [−140, −44] dBm (LTE/NR reporting range).
- Cut each trace into contiguous segments wherever rows were removed or the timestamp gap > 2 s;
  windows (inputs and all targets) never cross a segment boundary; RSRP_trend is reset at segment
  starts. Segments shorter than H + 5 are dropped.
- Same 12 input features (H = 15) as before; RSRP normalised as (r + 140)/96 without clipping;
  the other 11 MinMax-scaled on training traces only.

## Models and targets
- The six architectures of `../architectures.py`, unchanged except a 5-output head (k = 1…5 s).
- Two target types, trained separately: (A) absolute r_{t+k}; (B) delta r_{t+k} − r_t.
- Training config identical to `../config.py` (Adam 1e-3, wd 1e-5, batch 64, ≤100 epochs, early
  stop 20, LR plateau 0.5/10, clip 1.0, MSE, seed 42).

## Evaluation (test traces)
- Per k: MAE, RMSE, R² in dB; skill = 1 − MSE/MSE_persistence.
- Naive baselines: persistence r̂ = r_t; linear extrapolation r̂ = r_t + k(r_t − r_{t−1}).
- Also on the subset of windows where r_{t+k} ≠ r_t.
- Selected model: lowest validation MSE averaged over k (across both target types).
- Reported whether or not any model beats persistence.
