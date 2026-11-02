# Study 3: reliability-first use of forecasts (post hoc, evaluated on fresh scenarios)

**Written:** 2026-10-28, ~18:10, after the campaign agents (PREREGISTRATION-2 + AMENDMENT) finished training and **before any of their evaluation results were seen**. Not edited after Study 3 results are seen. **Labelled post hoc.** The pre-registered campaign is reported in full and unchanged.

## Why (training and calibration data only)
- **Training curves** (ε = 0.05): at the end of training the measurement-only agent has about 1 RLF per episode, against 24 for the oracle and 17–20 for the proactive agents.
- **Diagnostic on calibration scenario 2000** (seed-42 agents, greedy):

  | Agent | RLF | HO requests | Blocked by the admission check | Danger steps with an inadmissible target |
  |---|---|---|---|---|
  | Measurement-only | 2 | 649 | 25 | 0 |
  | Oracle | 29 | 837 | 405 | 150 |
  | Proactive (± fallback) | 25–27 | — | 32–88 | — |

- **Mechanism:** the forecast-based agents choose the target and compute the gain on forecasts even while the link is already in danger. The forecast-best cell often can't hold the link yet, so the handover is blocked, and the cell that is good *now* is never chosen.

## Change (one rule, forecast-based agents only)
**Reliability-first use of forecasts.** While a UE's link is in danger, i.e. its measured serving SINR < Q_out + δ_rel (the same condition that lifts the anti-oscillation limits), the UE uses the measured L3 RSRP as its signal source for state, target choice and gate, exactly as in the error fallback. Outside danger, nothing changes. Config flag: `danger_uses_measurement`.
- The measurement-only agent is unaffected by the rule.
- It is retrained anyway as the first arm of the lockstep, so that all four arms share the same random draws.
- It also serves as a reproducibility check against the campaign agent.

## Design
- **Everything else as in the campaign:** environment, layer, KPI-aligned reward (c_HO = 0.1), forecaster (CNN-LSTM, DEVIATIONS.md D1), τ_e = 5.66 dB, seeds 42–46, 100 episodes, arms.
- **Fresh evaluation scenarios, unused by any phase:**
  - headline 3100–3149 (50);
  - noise, mobility and speed sweeps 3100–3119.
- **Also evaluated on these scenarios:**
  - the baselines (fixed parameters);
  - the campaign's forecast-based agents (without the rule), labelled "pre-registered design".
- **Statistics:** KPIs, statistics and hypotheses H1–H5 as in PREREGISTRATION-2.md, plus **H6:** each forecast-based arm with the rule improves on the same arm without it (pre-registered design).
- **Outputs:** written to `logs/raw-3/` and `data/*-3/`.
