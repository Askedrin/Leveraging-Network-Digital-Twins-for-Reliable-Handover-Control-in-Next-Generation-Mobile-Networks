# v3 experiments: report (2026-10-28)

Everything below was produced in `ch5-outputs-v3/` and `retraining_pipeline/v3/`. Each phase has a pre-registration written before it ran. Post-hoc phases are labelled as such, and the deviations are listed in `DEVIATIONS.md`.

## 1. Sequence of studies

| Phase | Pre-registration | Question | Outcome | Report |
|---|---|---|---|---|
| 0 | `PREREGISTRATION.md` | Does perfect foresight help the original DQN design in a 3GPP TR 38.901 twin? | NO-GO, **degenerate**: every DQN learned "never hand over" (the reward had no failure terms; RLF re-attachment was free). | `PHASE0-REPORT.md` |
| 0b | `PREREGISTRATION-0b.md` (post hoc) | Same, with failure-aware reward and smooth spatial channel. | NO-GO: foresight gives ≤ 4 % and raises HOF. Diagnostic: the 30-s interval and blacklist cause almost all RLFs. | `PHASE0b-REPORT.md` |
| 0c | `PREREGISTRATION-0c.md` (post hoc) | Same, with a reliability-aware constraint layer (override when measured SINR < Q_out + 2 dB; target admission check). | NO-GO by criterion (oracle HOF up), but oracle −35 % RLF. The rule alone now beats A3; DQNs with the original shaped reward are far worse than the rule. | `PHASE0c-REPORT.md` |
| 1a | `retraining_pipeline/v3/PREREGISTRATION.md` | Do the six forecasters beat persistence on the real Cork traces? | Slightly, beyond 1 s (skill 0.08–0.14 at 3–5 s); not at 1 s (MAE). | §3 |
| 2 | `PREREGISTRATION-2.md` + `-AMENDMENT.md` (KPI-aligned reward, post hoc vs 0c) | Confirmatory campaign: 5 seeds, 50 fresh scenarios, MRO/tuned/static A3, sweeps. | Forecast-based agents: lower outage and better SINR, but 16–23 RLF/UE-h. Measurement-only DQN ≈ rule, both clearly better than every A3 variant. | §4 |
| 3 | `PREREGISTRATION-3.md` (post hoc, decided before Study 2 evaluation results) | Reliability-first use of forecasts (measurements while the link is in danger), on 50 new scenarios. | See §5. | §5 |

## 2. The simulated network (P12, P13, P18)
- **Configuration:** `config-0c/pinned-config.json`, with the reward from `config-2/reward.json` and the sources in `config/parameter-sources.md` and `config-0b/parameter-sources-0b.md`.
- **Layout:** `data/layout/cells.csv`, `render/layout.pdf`.
- **Radio statistics (scenarios 3000–3019, strongest-cell association):**
  - coupling loss: median 108 dB (5th–95th percentile 74–125 dB);
  - RSRP per RE: median −95 dBm;
  - SINR: median 11 dB;
  - the strongest cell changes 1,972 times per UE-hour.

  Sources: `data/twin-validation/`, `render/twin-cdfs.pdf`.
- **Volatility:** Static A3 makes 825 HO/UE-h (482 at 10–60 km/h). The Cork drive traces show 40–47 serving-cell changes per hour. The twin is a high-volatility scenario, and this is stated in the paper.

## 3. Forecasters (P4, P5)

**Real traces** (`retraining_pipeline/v3/results/summary.csv`, `selection.json`):
- persistence MAE is 1.08 / 2.58 / 3.14 dB at 1 / 3 / 5 s;
- no model beats it on MAE at 1 s;
- the best skill is 0.03 / 0.08 / 0.14;
- the selected model is BiLSTM with level target;
- the Transformer with level target was stopped unconverged (D2).

**Twin** (`data/predictor-twin/accuracy.csv`):
- at 1 s, persistence MAE is 4.85 dB;
- five architectures reach a skill of 0.32–0.35, BiLSTM 0.26;
- CNN-LSTM is used, tied with the Transformer (D1);
- τ_e = 5.66 dB.

## 4. Study 2: pre-registered campaign (scenarios 3000–3049; `data/headline/`)

| Arm | HO/UE-h | HOF | PP 1 s | RLF/UE-h [95 % CI] | Outage | 5th-pct SINR |
|---|---|---|---|---|---|---|
| Static A3 | 825 | 0.20 % | 8.1 % | 2.02 [1.66, 2.38] | 5.84 % | −9.44 dB |
| Tuned A3 (6 dB) | 622 | 0.19 % | 5.4 % | 4.91 [4.32, 5.53] | 6.50 % | −10.16 dB |
| MRO-A3 | 509 | 0.20 % | 3.5 % | 14.92 [13.90, 15.98] | 7.35 % | −10.86 dB |
| Rule only, original layer | 90 | 0.14 % | 0 | 297.0 | 11.05 % | −14.82 dB |
| Meas.-only DQN, original layer + reward (0b agents) | 67 | 0.46 % | 0 | 256.3 | 10.20 % | −13.85 dB |
| Meas.-only DQN, original reward (0c agents) | 120 | 0.50 % | 1.3 % | 217.3 | 9.52 % | −13.33 dB |
| Rule only, reliability-aware layer | 548 | 0.27 % | 4.8 % | 0.45 [0.29, 0.63] | 5.41 % | −8.45 dB |
| Measurement-only DQN | 545 | 0.24 % | 4.8 % | 0.49 [0.32, 0.67] | 5.43 % | −8.47 dB |
| Proactive DQN | 431 | 0.82 % | 3.4 % | 19.57 | 5.29 % | −8.31 dB |
| Proactive DQN + fallback | 447 | 0.76 % | 3.8 % | 16.29 | 5.30 % | −8.32 dB |
| Oracle DQN (1 s) | 347 | 2.27 % | 0.1 % | 23.40 | 5.15 % | −8.18 dB |

**Hypotheses** (`data/headline/hypotheses.json`): H1–H5 all false under the pre-registered definition, i.e. at least one decision KPI better with p_Holm < 0.05 and none significantly worse.
- **H1–H3:** the forecast-based arms have significantly more RLF and HOF than the measurement-only DQN and than every A3 variant. Against MRO-A3, RLF is +9 % (p = 0.03).
- **H4:** the measurement-only DQN and the rule are practically identical (0.49 vs 0.45 RLF/UE-h; outage 5.43 vs 5.41 %). The test flags these tiny but consistent differences, because the learning-arm metrics are averaged over 5 seeds per scenario.
- **H5:** the reliability-aware layer cuts RLF by 99.8 %, outage by 51 % and improves the 5th-percentile SINR by 6.4 dB. Its HOF rate is significantly higher (0.27 vs 0.14 %), which fails the strict criterion.

**Reported, not hypotheses:** the measurement-only DQN vs Static A3 gives RLF −76 %, outage −7 %, 5th-percentile SINR +1.0 dB, 34 % fewer handovers and fewer ping-pongs (all p_Holm < 10⁻⁶), with HOF n.s.

**Seed variability** (`seed_std.csv`): the standard deviation of RLF/UE-h across seeds is 0.06–0.27.

**Sweeps** (`data/robustness/`, `data/mobility/`, `data/speed/`):
- **Noise:**
  - the error fallback works as designed: as σ grows it engages more (6 % → 99 % of UE-steps) and brings RLF back to the measurement-only level (σ = 8 dB: 0.50 vs 19.7 without fallback);
  - the outage of the proactive agent without fallback rises to 5.9 %.
- **Mobility:** the ranking is the same under the four mobility models.
- **Speed:**
  - at 10–60 km/h, Static A3 keeps a better 5th-percentile SINR (−3.6 vs −5.2 dB) than the rule and the DQN, at 57 % more handovers;
  - its RLF is 0.86 vs 0.64.

**Sensitivity (D1, `data/sensitivity/`):** the agents (trained with the CNN-LSTM's forecasts) are evaluated with the Transformer's forecasts, without retraining.
- RLF rises from 16.3 to 25.2 per UE-h with fallback, and from 19.6 to 30.1 without.
- Outage is +2 % and the 5th-percentile SINR −0.12 dB (all p_Holm < 10⁻¹²).
- Fallback engagement is 6.9 % against 6.2 %.

In this design, where forecasts are also used in danger, swapping the forecaster hurts. Keeping the CNN-LSTM did not make the forecast arms look worse than the rule's choice would have.

## 5. Study 3: reliability-first use of forecasts (scenarios 3100–3149; `data/*-3/`)

Same agents, layer, reward and forecaster as Study 2, except that a UE in danger (measured serving SINR < Q_out + δ_rel) uses measurements (`PREREGISTRATION-3.md`). All arms were retrained (5 seeds) and evaluated on 50 scenarios that no earlier study used. The Study 2 design is re-evaluated on the same scenarios (`prereg-ref`). These are the numbers in the paper.

| Arm | HO/UE-h | HOF | PP 1 s | RLF/UE-h [95 % CI] | Outage | 5th-pct SINR | Fallback |
|---|---|---|---|---|---|---|---|
| Static A3 | 818 | 0.17 % | 7.9 % | 1.94 [1.60, 2.30] | 5.76 % | −9.29 dB | |
| Tuned A3 (6 dB) | 618 | 0.17 % | 5.2 % | 5.04 [4.45, 5.65] | 6.39 % | −10.00 dB | |
| MRO-A3 | 506 | 0.17 % | 3.4 % | 14.87 [13.68, 16.04] | 7.20 % | −10.72 dB | |
| Rule only, original layer | 90 | 0.22 % | 0 | 296.95 [290.93, 302.98] | 11.00 % | −14.83 dB | |
| Rule only, reliability-aware layer | 541 | 0.18 % | 4.6 % | 0.67 [0.49, 0.85] | 5.34 % | −8.36 dB | |
| Measurement-only DQN | 537 | 0.16 % | 4.5 % | 0.74 [0.56, 0.93] | 5.35 % | −8.38 dB | |
| Proactive DQN | 535 | 0.18 % | 4.5 % | 0.71 [0.53, 0.90] | 5.32 % | −8.34 dB | |
| **Proactive DQN + fallback (proposed)** | 535 | 0.17 % | 4.4 % | 0.70 [0.52, 0.89] | 5.32 % | −8.34 dB | 6.2 % |
| Oracle DQN (1 s) | 532 | 0.19 % | 4.5 % | 0.68 [0.50, 0.87] | 5.28 % | −8.30 dB | |
| Proactive DQN, forecasts also in danger | 426 | 0.76 % | 3.5 % | 19.17 [17.91, 20.49] | 5.20 % | −8.19 dB | |
| Proactive DQN + fallback, forecasts also in danger | 443 | 0.72 % | 3.7 % | 15.81 [14.75, 16.92] | 5.21 % | −8.21 dB | 6.2 % |
| Oracle DQN, forecasts also in danger | 347 | 2.07 % | 0.1 % | 22.53 [21.09, 24.00] | 5.06 % | −8.06 dB | |

The UE is in danger in 2.9 % of the UE-steps.

**Hypotheses** (`data/headline-3/hypotheses.json`, same definition as §4):
- **H3 true against all three A3 variants.** Against static A3: RLF −64 %, outage −7.6 %, 5th-percentile SINR +0.95 dB (all p_Holm < 10⁻⁵), HOF n.s., 35 % fewer handovers.
- **H5 true.** The reliability-aware layer vs the original layer, rule only: RLF −99.8 %, outage −51 %, +6.5 dB; HOF n.s. this time.
- **H1, H2 false.** The forecast agents are better than the measurement-only DQN in RLF (−5 % with fallback, −4 % without), outage (−0.6 %) and 5th-percentile SINR (+0.03 dB), but have a significantly higher HOF rate (+5 % and +11 %).
- **H4 false.** The rule is *better* than the measurement-only DQN in RLF (0.67 vs 0.74), outage and 5th-percentile SINR; the DQN has the lower HOF rate (0.16 vs 0.18 %).
- **H6 false.** Reliability-first use vs the Study 2 design: RLF −96 %, HOF −77 %, but outage +2 % and 5th-percentile SINR −0.13 dB (significant), so the strict criterion fails.

**Reported, not hypotheses:**
- **Oracle vs measurement-only DQN:** RLF −8 %, outage −1.3 %, +0.08 dB; HOF +20 %.
- **Fallback vs no fallback:** no significant difference with clean forecasts.
- **Proposed vs rule only:** RLF +5 % (worse), outage −0.3 %, +0.02 dB (better), HOF n.s. The oracle vs the rule shows the same pattern (RLF +2 %, outage −1 %), with HOF +7 %. No learned agent dominates the rule.

**Seed variability:** the standard deviation of RLF/UE-h across seeds is 0.03–0.08.

**Sweeps** (`data/robustness-3/`, `data/mobility-3/`, `data/speed-3/`; 20 scenarios each):
- **Noise:** up to σ = 8 dB the proposed agent's RLF changes by < 6 % and its outage by < 1 %. The fallback engages in 6 % → 99 % of UE-steps, but there is little left to protect.
- **Mobility:** the same ranking under all four models. Against static A3, outage and 5th-percentile SINR are significantly better everywhere. RLF is lower everywhere but significant only on the Manhattan grid and under the random walk. The rule has the lowest (or tied) RLF in every condition.
- **Speed:**
  - at 100–130 km/h, RLF −74 % vs static A3;
  - at 10–60 and 30–60 km/h, 11–12 % less outage and 37–38 % fewer HOs, RLF n.s., but static A3 has a better 5th-percentile SINR (−3.5 vs −5.2 dB at 10–60 km/h).

**Sensitivity (D1, `data/sensitivity-3/`):** with the Transformer's forecasts (agents not retrained), the proposed agent gets RLF 0.71 vs 0.70 per UE-h, with outage and 5th-percentile SINR unchanged; nothing is significant (p_Holm ≥ 0.09).

**Latency (P11, `data/latency/`):** one decision for one UE on one core takes 0.96 ms (p50), 1.03 ms (p95), 1.17 ms (p99). The forecaster takes 0.85 ms, the Q-network 0.11 ms, the constraint layer 5 µs. The Transformer forecaster alone takes 44 ms (`latency-Transformer.csv`).

## 6. Other outputs
- **P10, training curves:** `data/training*/`, `render/training*.pdf`.
- **P11, latency:** `data/latency/` (Study 3 agent, scenario 3100; see §5).
- **Serving RSRP over time:** `data/rsrp-trace-3/` (static A3 and the proposed agent, 5 seeds, 50 headline scenarios). The per-scenario means reproduce `mean_rsrp_dbm` of the evaluation exactly; overall −93.8 vs −94.0 dBm.

## 7. Runtimes (CPU: Intel Xeon w3-2435, 16 threads; jobs ran concurrently)

| Job | Runtime |
|---|---|
| Phase 0c training | ≈ 57 min per seed (3 seeds in parallel, 4 threads) |
| Twin forecasters | 4–150 min each, 1 thread |
| Study 2 training | ≈ 100 min per seed (5 in parallel, 2 threads) |
| Study 2 evaluation and sweeps | ≈ 113 min per seed |
| Baselines, all conditions | 4 min |
| MRO warm-up and selection | ≈ 10 min |
| Real-trace forecasters | 1–13 h each, under contention |
| Study 3 training | ≈ 177 min per seed (5 in parallel, 2 threads) |
| Study 3 evaluation and sweeps | ≈ 69 min per seed |
| Transformer forecasts for D1 | ≈ 70 min per 50 scenarios (5 processes, 3 threads) |
| Latency benchmark | < 1 min |

## 8. Reproduction (from `ch5-outputs-v3/scripts/`)
```
python calibrate.py --phase 0c                     # override margin, tuned A3
python campaign.py mro                             # MRO warm-up and step
python reward_dev.py --ho-cost 0 ; python reward_dev.py --ho-cost 0.1 ; python reward_dev.py --select
python forecaster_twin.py data
for a in ImprovedLSTM TCN CNNLSTM LSTMAttention BiLSTM Transformer; do python forecaster_twin.py train --arch $a; done
python forecaster_twin.py select                   # then keep CNN-LSTM as in DEVIATIONS.md D1
python campaign.py baselines [--study 3]
for s in 42 43 44 45 46; do python campaign.py train --seed $s [--study 3]; done
for s in 42 43 44; do python campaign.py reference --phase 0b --seed $s; python campaign.py reference --phase 0c --seed $s; done
for s in 42 43 44 45 46; do python campaign.py prereg-ref --study 3 --seed $s; done
python campaign.py sens-forecasts [--study 3]; for s in 42 43 44 45 46; do python campaign.py sensitivity --seed $s [--study 3]; done
python aggregate_campaign.py --study 2 ; python aggregate_campaign.py --study 3
python twin_validation.py ; python benchmark_latency.py ; python benchmark_latency.py --arch Transformer
for i in 0 1 2 3 4 5 6 7; do python rsrp_trace.py --part $i --parts 8; done ; python rsrp_trace.py --merge
python figures.py ; STUDY_SUFFIX=-3 python figures.py ; python tables.py
```
The paper's `figs/` are copies of `render/`: `ablation.pdf` and `training.pdf` come from `render/ablation-3.pdf` and `render/training-3.pdf` (Study 3); the other figures have the same names.

Four figures redraw the submitted paper's result figures in their original form with the new data:
- `forecast-real` (LSTM-Attention and TCN vs measured RSRP);
- `training` (reward convergence of the proposed agent);
- `rsrp-time` (mean serving RSRP over time, data from `rsrp_trace.py`);
- `kpi` (ping-pong, RLF and total handovers vs static A3; HSR is left out because it is identical, 99.83 %, and stays in the main table).

`ablation` is a single RLF bar chart, using the paper's method names. `tables.py` writes `res-main`, `res-forecast` (RMSE, real and simulated) and `res-mobility` (A3 vs proposed, * = significant).
Real traces: `cd retraining_pipeline/v3 && python prep.py && python train_eval.py --arch <A> --target <abs|delta> && python summarize.py`.

## 9. Disagreements and surprises
- **The original design fails in four distinct ways once the channel no longer contains the LSTM.**
  1. The reward has no failure terms, so the agent never hands over.
  2. The 30-s interval and the blacklist cause RLFs in a volatile channel.
  3. The shaped reward's recency penalties make failing cheaper than following the best cell.
  4. Using forecasts while the link is in danger causes RLFs.

  Each was diagnosed on training or calibration data and fixed in a labelled post-hoc step.
- **The rule alone is at least as reliable as every learned agent.** In Study 3 it has the lowest RLF of all arms, the oracle included, and the agents gain only fractions of a percent in outage. Under the reliability-aware layer, the layer itself is the reliability mechanism.
- **A more accurate forecaster is not a better one for control.** The Transformer has a lower validation MSE, but its forecasts raise RLF by 55 % in the Study 2 design and change nothing in Study 3.
- **The forecaster is far more useful in the twin than on real traces.** The twin's skill (0.34) is likely optimistic.
- **The twin's handover rate is about 18× the operational network's.**
- **The tuned and MRO A3 baselines trade RLF for fewer ping-pongs** under the 1 : 1 : 0.1 objective.
