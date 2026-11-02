# Deviations from the pre-registrations

## D1. In-twin forecaster: CNN-LSTM kept although the Transformer ended with a 0.3 % lower validation MSE
**Recorded:** 2026-10-28, ~17:40, after the Transformer's epoch 22. At this point the campaign agents were still training; no Phase 2 learning-arm evaluation result existed.

**Rule (PREREGISTRATION-2.md §1):** select the architecture with the lowest validation MSE.

**What happened:**
- **Provisional start.** To use idle CPU, the campaign was started when 5 of 6 architectures had finished. CNN-LSTM had the lowest validation MSE (0.0035252, normalised); `selection.json` is marked `"complete": false`.
- **The Transformer overtook it.** It was still training. At epoch 22 it reached 0.0035214, 0.1 % lower, and it may improve further before early stopping.

**Decision:** keep CNN-LSTM for the campaign.
1. **The difference is not meaningful.** 0.1 % is far below the epoch-to-epoch fluctuation of either model's validation MSE (2–5 % between consecutive epochs), so the two are tied.
2. **The Transformer is much more expensive.** It has 1.93 M parameters against 0.19 M; a full-scenario forecast takes about 90 s against about 10 s on this CPU. Re-running the campaign would take about 7 h, and the paper reports per-decision latency (P11).

**Final numbers:** after its last epoch (30), the Transformer's validation MSE is 0.0035137, against CNN-LSTM's 0.0035252 (0.3 % lower). Under the rule, τ_e would have been 5.75 dB instead of 5.66 dB (`data/predictor-twin/selection-rule-transformer.json`). Test skill: 0.346 against 0.342.

**Mitigation (sensitivity check, run after training):** the trained agents of every seed are also evaluated on the 50 headline scenarios with the *Transformer's* forecasts (`condition = "sensitivity"`, `variant = "transformer-forecasts"`). The realised error and τ_e are unchanged. Reported next to the headline.

**Result of the sensitivity check (run 2026-10-28/29).** The check was extended to Study 3, the design reported in the paper. Agents were trained with the CNN-LSTM's forecasts and evaluated with the Transformer's, without retraining:
- **Study 3** (`data/sensitivity-3/`, scenarios 3100–3149): proposed agent RLF 0.71 vs 0.70 per UE-h; outage 5.33 % vs 5.32 %; 5th-percentile SINR −8.34 dB both. No KPI differs significantly (p_Holm ≥ 0.09). Against the measurement-only DQN the signs are unchanged.
- **Study 2** (`data/sensitivity/`, scenarios 3000–3049; forecasts also used in danger): RLF 25.2 vs 16.3 per UE-h with fallback, 30.1 vs 19.6 without. The Transformer's forecasts are worse here.
- **Fallback engagement:** 6.9 % vs 6.2 % of UE-steps in both studies.

**Measured per-decision cost** (`data/latency/`, batch 1, one core): Transformer 44 ms against CNN-LSTM 0.85 ms.

**Conclusion:** the deviation does not favour the reported results.

**In the paper:** the selection is stated as a tie, resolved by inference cost, and this deviation is disclosed. §V-E reports the sensitivity result and §V-H the latency.

## D2. Real-trace forecaster: Transformer with level target stopped before convergence
**Recorded:** 2026-10-28, ~18:40.

**Rule (retraining_pipeline/v3/PREREGISTRATION.md):** train each model up to 100 epochs with early stopping (patience 20).

**What happened:** under CPU contention with the twin experiments, the last of the 12 real-trace models (Transformer, level target) had run for 13.3 h and reached about epoch 36 of 100. Its logged validation MSE (normalised level target, logged every 5 epochs) was:

| Epoch | 1 | 6 | 11 | 16 | 21 | 26 | 31 | 36 |
|---|---|---|---|---|---|---|---|---|
| Validation MSE | 0.00310 | 0.00280 | 0.00219 | **0.00192** | 0.00214 | 0.00207 | 0.00243 | 0.00264 |

The best, about 17.7 dB² in dB², is above the selected model's validation MSE (BiLSTM, level target, 16.95 dB²), and the trend after epoch 16 is upward.

**Decision:** stop it; it has no checkpoint and no test result. In the paper's table it is listed as not converged within the compute budget. The selection among the 11 completed models is unaffected unless this model's unlogged best fell below 16.95 dB², which the logged trajectory makes very unlikely.
