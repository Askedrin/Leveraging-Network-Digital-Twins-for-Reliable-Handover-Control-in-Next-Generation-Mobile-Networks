# Phase 0c: pre-registration (post hoc relative to Phase 0b)

**Written:** 2026-10-28, after Phase 0b (`PHASE0b-REPORT.md`) and before any Phase 0c calibration or training run. Not edited after results are seen.

**Approval:** after reading the Phase 0b report, the authors delegated the choice among options A/B/C to us ("go with what you see right and get us what we need", 2026-10-28). We chose option B.

**Labelling:** post hoc, and the third iteration of the method after Phase 0 and 0b. It is motivated by the mechanism diagnosed in PHASE0b-REPORT.md §3 (the minimum interval and the blacklist block handovers the link needs; the oracle hands over to cells that can't yet hold the link), not by a KPI target. The defaults of `TwinConfig` still reproduce Phase 0. The Phase 0b settings are reproduced by `config-0b/`.

## Changes from Phase 0b (and only these): a reliability-aware constraint layer

| # | Change | Why |
|---|---|---|
| 1 | **Reliability override.** For UE *v* at step *t*: if its *measured* serving SINR (L3-filtered, the value the network gets from SS-SINR reports) is below Q_out + δ_rel, then the minimum handover interval and the blacklist don't block a handover at that step, both in the agent's forced-stay mask and in the execution gate. The 3-dB gain condition stays. Because the layer allows the move, the reward's blacklist penalty isn't applied to it. | The anti-ping-pong limits cap the handover rate at about 100/UE-h, while this channel needs 800–1,000. With both limits removed, the rule-only policy has 0 RLF (PHASE0b-REPORT.md §3). Keeping the limits outside link danger preserves their ping-pong role. |
| 2 | **Target admission check.** A handover through the gate executes only if the target's *current* measured SINR ≥ Q_in (−6 dB). This applies to every gated arm (DQN arms, Rules only), not to the A3 baselines, which stay plain TS 38.331 A3. | The oracle arms handed over to cells that would be better in 1–2 s but couldn't hold the link at execution (HOF 8–17 %). A network would not command a handover to a target that the UE currently reports as unusable. |

**δ_rel selection** (before any learning arm runs):
- Rules-only policy with changes 1 and 2, on calibration seeds 2000–2004.
- Grid δ_rel ∈ {0, 2, 4, 6, 8} dB.
- Minimise (N_RLF + N_HOF + 0.1·N_PP,1s) per UE-hour, the Phase 0b tuned-A3 objective. Ties go to fewer handovers.
- Recorded in `config-0c/override-grid.csv` and `config-0c/pinned-config.json`.

## Unchanged from Phase 0b
- **Environment:** channel v2, failure-aware reward and re-establishment semantics.
- **Agents:** arms (Reactive, Oracle k = 5, Oracle k = 10), hyperparameters and state.
- **Mobility and seeds:** mobility, training seeds 42/43/44, 100 episodes.
- **Evaluation and statistics:** evaluation scenarios 1000–1019, KPIs, statistics.
- **Baselines:** Static A3 and Tuned A3 parameters (`config-0b/baselines.json`; A3 doesn't use the constraint layer).
- **Go/no-go criterion:** PREREGISTRATION.md.
- **Degeneracy rule:** PREREGISTRATION-0b.md.

## Also reported (not part of the criterion)
- Rules only and Reactive DQN, Phase 0b layer vs Phase 0c layer.
- Rules only vs Reactive DQN.
- Static A3 vs every gated arm.

## What happens after Phase 0c (decided now, before results)
- **The confirmatory campaign runs whatever the verdict** (Phase 1b: in-twin forecaster, proactive arm, error-based fallback; Phase 2: ≥ 5 seeds, baselines, sweeps), under its own pre-registration (`PREREGISTRATION-2.md`), written before it runs.
  - **GO:** the paper keeps "proactive" in its title and claims.
  - **NO-GO:** the paper is framed as a reliability-aware coupling with an oracle-bounded (small) value of foresight. The proactive arm is then reported as an ablation.
- **Fresh evaluation scenarios.** Scenarios 1000–1019 have been seen in three phases, so the Phase 2 headline uses new evaluation scenarios that no phase has used.
