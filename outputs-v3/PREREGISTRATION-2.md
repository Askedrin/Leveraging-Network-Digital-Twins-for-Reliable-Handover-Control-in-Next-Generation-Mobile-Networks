# Phase 1b + Phase 2: pre-registration of the confirmatory campaign

**Written:** 2026-10-28, while Phase 0c was training and before its results or any Phase 1b/2 run. Not edited after results are seen. Anything decided later is labelled post hoc in the report.

**Approval:** standing delegation by the authors (2026-10-28). PREREGISTRATION-0c.md fixed that this campaign runs whatever the Phase 0c verdict is.

**Common to all runs:**
- channel v2;
- failure-aware reward and re-establishment semantics (Phase 0b);
- the **Phase 0c reliability-aware constraint layer**, with δ_rel = 2 dB selected in `config-0c/override-grid.csv`, for every gated arm;
- Manhattan mobility at 60–100 km/h for training;
- 20 vehicles, 1,000 steps of 0.2 s per episode.

Config: `config-2/pinned-config.json`.

## 1. Phase 1b: the in-twin basic model (P5)

| Item | Pre-registered choice |
|---|---|
| Data ("UDR" of the twin) | L3-filtered RSRP histories of every UE–cell link in dedicated forecaster scenarios: seeds 8,000,000–8,000,019 (train), 8,000,020–024 (validation), 8,000,025–029 (test). These seeds are disjoint from the agent training, calibration and evaluation seeds. Measurements are policy-independent, so no agent is involved. |
| Windows | History H = 15 steps (3 s). Candidate links only (L3 RSRP ≥ r_min = −125 dBm at the issue time). 3,000 windows sampled uniformly per scenario (fixed seed), giving 60,000 / 15,000 / 15,000. |
| Inputs (12, the paper's 12-feature contract reinterpreted for the twin) | x, y (normalised by the grid); speed/100; L3 RSRP of the link, (r + 140)/96; distance to that cell/2000; sin and cos of the heading; cos and sin of the angle between the heading and the bearing to the cell (these replace hour sin/cos, which are constant in the twin); RSRP trend (r_t − r_{t−1})/10; cell type (0, 0.5, 1); change of distance per step/(speed·Δt). "Handover_recent" is dropped because it depends on the policy. |
| Target | Direct k = 5 (1 s) delta of the link's L3 RSRP, (r_{t+5} − r_t)/96. k = 5 is the horizon of the submitted paper. |
| Models | The six architectures of `retraining_pipeline/architectures.py`, unchanged. Optimiser, learning rate, batch size and gradient clipping from `retraining_pipeline/config.py`. **Changed for the CPU budget:** at most 30 epochs, early-stop patience 5, learning-rate patience 3. |
| Selection | Lowest validation MSE. |
| Reported | Test MAE, RMSE and R² in dB, and the skill vs persistence (1 − MSE/MSE_persistence). Linear-extrapolation baseline r_t + 5(r_t − r_{t−1}). The same metrics on the windows where the target changes by more than 1 dB. |

**Realised forecast error (the NDT feedback, s₁₃).**
- For UE *v* at step *t*: e_t(v) is the mean, over the last W = 25 steps (5 s), of the mean absolute difference between the forecast issued 5 steps earlier and the L3 RSRP now. The difference is averaged over the UE's candidate cells and uses the forecast the arm actually received, including any injected noise.
- s₁₃ = min(e_t/10 dB, 1) for the proactive arms. It is 0 for all other arms, as in Phase 0.

**Fallback threshold:** τ_e = the 95th percentile of e_t with clean forecasts on the forecaster validation scenarios, computed before any agent is trained.

## 2. Phase 2: arms

**Learning arms** (trained in lockstep with the shared-RNG procedure of Phase 0; seeds **42, 43, 44, 45, 46**; 100 episodes each):

| Arm | Signal source x for the state and the target choice |
|---|---|
| Reactive DQN | L3 RSRP at t (= a persistence forecast) |
| Oracle DQN k = 5 | true future L3 RSRP at t + 5 (upper bound) |
| Proactive DQN | selected forecaster's prediction of the L3 RSRP at t + 5 |
| Proactive DQN + fallback | as Proactive, but a UE with e_t > τ_e uses L3 RSRP at t |

Oracle k = 10 is dropped to bound compute: k = 5 is the paper's horizon, and the 2-s oracle added nothing in Phase 0b.

**Baselines:**

| Arm | Definition |
|---|---|
| Static A3 | margin 3 dB, TTT 160 ms (`config-0b/baselines.json`) |
| Tuned A3 | margin 6 dB, TTT 40 ms (`config-0c/baselines.json`) |
| MRO-A3 | See below. |
| Rules only | the 3-dB rule behind the Phase 0c constraint layer |
| Rules only, paper's layer | the 3-dB rule behind the submitted paper's layer (30 s interval + blacklist, no override, no admission check); decomposition only |

**MRO-A3:**
- A3 with a per cell-pair individual offset: it fires when M_n + CIO[s,n] > M_s + 3 dB for 160 ms.
- CIO is updated from the TS 38.300 §15.5.2 failure classes, with step Δ and clipping to [−6, +6] dB:
  - **too late** (RLF in s, no HO within T_store = 5 s, re-establishment in n): CIO[s,n] += Δ;
  - **too early** (RLF ≤ 5 s after HO s→n, re-establishment in s): CIO[s,n] −= Δ;
  - **wrong cell** (RLF ≤ 5 s after HO s→n, re-establishment in m): CIO[s,n] −= Δ and CIO[s,m] += Δ;
  - **1-s ping-pong** s→n→s: CIO[s,n] and CIO[n,s] each −= Δ/2.
- CIOs start at 0 and adapt over 50 warm-up scenarios (seeds 9,000,000–9,000,049), carried over from one scenario to the next, and are then frozen for evaluation.
- Δ ∈ {0.25, 0.5, 1.0} dB is selected on the calibration seeds 2000–2004 after warm-up, with the Tuned-A3 objective (N_RLF + N_HOF + 0.1·N_PP,1s per UE-h).

**PPO (Voigt et al., 2025) is not included.** A faithful port needs their simulator's state, action and reward definitions. A re-implementation in this twin would be a different method, so we don't ship a "PPO-like" baseline and state this as a limitation.

**Reported for reference** (not part of the hypotheses): the Phase 0b agents (3 seeds, paper's layer, Reactive and Oracle k = 5) evaluated on the new scenarios.

## 3. Evaluation and statistics
- **Headline (P1):** 50 new evaluation scenarios, seeds 3000–3049, never used before.
  - Learning-arm KPIs are averaged over the 5 training seeds per scenario.
  - Reported as mean with a 95 % bootstrap CI over scenarios (10,000 resamples).
- **KPIs:** HO/UE-h, HOF rate, 1-s and 10-s ping-pong rates, RLF/UE-h (with the MRO classes), outage fraction, mean and 5th-percentile SINR, mean RSRP.
- **Tests:** paired Wilcoxon signed-rank across the 50 scenarios, with Holm correction over the 4 decision KPIs (RLF/UE-h, HOF rate, outage fraction, 5th-percentile SINR) within each comparison.

**Hypotheses** (each "improves on" means at least one decision KPI better with p_Holm < 0.05 and none significantly worse):

| | Hypothesis |
|---|---|
| H1 (value of the coupling) | Proactive + fallback improves on Reactive. |
| H2 (value of the forecast) | Proactive improves on Reactive. |
| H3 (reliability vs 3GPP) | Proactive + fallback improves on Static A3, on Tuned A3 and on MRO-A3; HO rate and ping-pong are reported alongside. |
| H4 (value of learning) | Reactive DQN improves on Rules only (same layer). |
| H5 (value of the constraint-layer change) | Rules only improves on Rules only with the paper's layer. |

## 4. Sweeps (evaluation only; the same trained agents; scenarios 3000–3019)
- **P6, forecast robustness:**
  - i.i.d. Gaussian noise σ ∈ {0, 1, 2, 3, 5, 8} dB added to all forecasts;
  - plus one bias case, +3 dB on all non-serving forecasts (σ = 0);
  - arms Proactive and Proactive + fallback, with Reactive as the reference.
  - Reported per σ: the 4 decision KPIs, the fraction of UE-steps in fallback, and paired tests vs Reactive (Holm over the 4 KPIs per σ).
- **P7, mobility:** Manhattan, random walk (v2), Random Waypoint (no pause) and Gauss-Markov (α = 0.75 per 1-s update, σ_heading = 20°, σ_speed = 5 km/h), all at 60–100 km/h. All arms; the forecaster is not retrained.
- **P8, speed (Manhattan):**
  - 10–60 km/h (the 5th–95th percentile of the moving speeds in the Cork traces);
  - 30–60 km/h;
  - 60–100 km/h;
  - 100–130 km/h.

  All arms. Static A3's HO rate per band is reported next to the Cork reference of 40–47 cell changes per hour.

## 5. Other outputs
- **P10:** training curves of every arm and seed.
- **P11:** batch-1 latency (p50/p95/p99) of the forecaster, the agent, the constraint layer and end to end, with the hardware.
- **P13:** layout.
- **P18:** CDFs of coupling loss, RSRP and serving SINR (Static A3 association) on scenarios 3000–3019.

## 6. Rules
All results are reported, including those against the paper. There is no seed picking, no reruns for a better draw, and no change to the reward, hyperparameters, δ_rel, τ_e or the forecaster after evaluation results are seen.
