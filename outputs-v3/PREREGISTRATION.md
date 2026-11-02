# v3 Phase 0 — pre-registration (go/no-go test)

Written 2026-10-27, **before any v3 evaluation run**. Not edited after results are seen; anything
decided later is labelled post hoc in REPORT/PHASE0-REPORT. Brief: `../resubmission/HANDOFF-2-EXPERIMENTS.md`.

## Question
Does foresight of the radio channel improve the DQN handover agent at all, when the agent is
given **perfect** future measurements? If even an oracle does not help, a learned forecaster cannot.

## Environment (v3 simulated network) — what changes from v2
- **No learned model in the channel.** Ground truth = TR 38.901 path loss + LOS probability (UMa for
  macro, UMi-Street Canyon for micro/pico) + spatially correlated shadowing; LOS state spatially
  consistent (correlated Gaussian mapped through Φ, compared with Pr_LOS). Values and sources:
  `config/parameter-sources.md`.
- RSRP per resource element (Tx power − 10log10(12·N_RB) + gain − PL − SF). Measurements = truth +
  residual fading N(0, σ_F=1 dB), Layer-3 filtered (k=4).
- SINR = S / (Σ η_c I_c + N) with cell load factors η_c.
- Mobility: Manhattan grid (Phase 0). Layout: v2's 12 cells (`make_cells`, unchanged).
- 20 vehicles, Δt = 0.2 s, 1,000 steps (200 s) per episode.

## KPIs (identical for all arms; replace v2's)
- HOF: target-cell true SINR < Q_out = −8 dB at execution → failure; UE re-establishes on the
  strongest cell. HSR = 1 − N_HOF/N_HO.
- RLF: serving true SINR < Q_out for T310 = 1 s (5 consecutive steps; timer stops if SINR > Q_in =
  −6 dB). UE re-establishes on the strongest cell. Classified too-late / too-early / wrong-cell /
  coverage using T_store = 5 s.
- Ping-pong: return to the previous source cell with time-of-stay < 1 s (primary) and < 10 s
  (secondary).
- Outage fraction (UE-steps with serving true SINR < Q_out), RLF per UE-hour, HO per UE-hour,
  mean and 5th-percentile serving true SINR (per scenario), mean serving true RSRP.

## Arms
| Arm | Signal source x (state s2–s5, s16, s17; target selection; execution gate) |
|---|---|
| Reactive DQN | current L3 measurement |
| Oracle DQN k=5 (1 s) | L3 measurement at t+5 (precomputed, policy-independent) |
| Oracle DQN k=10 (2 s) | L3 measurement at t+10 |
| Rules only | constraint layer, always request strongest current neighbour |
| Static A3 | offset 0 dB + hysteresis 3 dB, TTT 160 ms (fires on the 2nd consecutive 0.2 s sample) |
| Tuned A3 | margin ∈ {0..6} dB × TTT ∈ {40, 80, 160, 320, 640} ms, chosen on calibration seeds by minimum (N_RLF + N_HOF + N_PP,1s) per UE-hour, ties → fewer HOs |

DQN network, hyperparameters, reward (`shaped_reward`), constraint-layer rules: **unchanged from
v2** (imported from `ch5-outputs-v2/scripts/common.py`). State: v3 definition (manuscript Table IV):
neighbours exclude the serving cell; s12 = HOs in last 60 s; s13 = 0 (no forecaster in Phase 0);
s15 = measured serving SINR; RSRP normalisation (r+140)/80.
**Execution gate:** gain ≥ 3 dB measured in the arm's own source x (for Reactive this is identical
to v2; for oracle arms it is the future gain — otherwise the gate would block every early handover).

## Calibration (A3 and Rules only; seeds 2000–2004; before any learning arm runs)
Default: macro/micro/pico Tx 49/37/30 dBm, speeds 60–100 km/h. Accept if Static A3 has
(N_HOF + N_RLF)/N_HO between 1 % and 10 % (mean over the 5 seeds). If below 1 %: lower all Tx
powers by 3 dB and retry (max −9 dB); if above 10 %: raise by 3 dB (max +6 dB). Speed band is not
changed in Phase 0. Reward rescaling only if > 50 % of serving true RSRP samples fall below
−120 dBm under Static A3. The calibration log is kept.

## Protocol
- Training seeds 42, 43, 44; 100 episodes each; episode scenario seed = seed·10^6 + episode.
- Evaluation: scenarios 1000–1019, identical for all arms and training seeds.
- Learning-arm metrics averaged over the 3 training seeds per scenario, then paired Wilcoxon
  signed-rank across the 20 scenarios; Holm correction over the 4 decision KPIs, separately per
  oracle arm.

## Go / no-go criterion
**GO** if Oracle k=5 **or** Oracle k=10 beats Reactive on at least one of
{RLF per UE-hour, HOF rate, outage fraction, 5th-percentile SINR} with Holm-adjusted p < 0.05 and a
relative improvement ≥ 10 %, with no significant (Holm p < 0.05) degradation of the other three.
**NO-GO** otherwise. Also reported (not part of the criterion): Rules-only vs DQN, Static vs Tuned A3.

## Not done
No seed selection, no reruns for a better draw, no change of reward/hyperparameters after results,
no tuning on evaluation scenarios. STOP after Phase 0 and report to the user.
