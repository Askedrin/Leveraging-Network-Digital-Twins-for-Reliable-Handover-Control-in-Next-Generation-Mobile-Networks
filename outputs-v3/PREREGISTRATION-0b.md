# Phase 0b: pre-registration (post hoc relative to Phase 0)

**Written:** 2026-10-27, after Phase 0 (`PHASE0-REPORT.md`) and before any Phase 0b calibration or training run. Not edited after results are seen.

**Approval:** the authors chose "Option B" on 2026-10-27, i.e. fix the method so that it counts link failures, and re-test.

**Labelling:** Phase 0b is labelled **post hoc**. It was designed after seeing that Phase 0 was degenerate (every DQN learned "never hand over"). Phase 0 and its code paths stay reproducible: the defaults of `TwinConfig` reproduce Phase 0 exactly (checked on scenario 1000).

## Changes from Phase 0 (and only these)

| # | Change | Why |
|---|---|---|
| 1 | **Failure-aware reward.** The v2 reward stays unchanged, plus: −1 per step in which the serving true SINR < Q_out; −15 per RLF; −15 per handover failure. Outage and RLF penalties are credited to the action taken at the previous step, i.e. the action that led to them; the HOF penalty goes to the executing action. | Phase 0 reward had no term for what the KPIs measure. Scale: one failure (≈ 5 outage steps + RLF) costs ≈ −20, the same as the reward's own heaviest ping-pong penalty (−20). The total stays inside the replay clip [−50, 10]. |
| 2 | **RLF re-establishment counts as a change of serving cell:** it resets the time since change used by the reward and by state s₁₁. It is not counted as a handover (N_HO, blacklist and constraint layer unchanged). | In Phase 0, failing was a free handover that kept the long-stay bonus. |
| 3 | **Channel version 2:** smooth (Matérn-3/2, cascaded AR(1)) spatial processes for the LOS variable and for shadowing, keeping the TR 38.901 1/e correlation distances; one shadowing process per link, scaled by the current LOS state. | Phase 0's first-order processes are rough at the 4–6 m step scale (LOS flicker, 3.4 dB per-step relative shadowing changes), and the separate LOS/NLOS shadowing processes jumped at every LOS change. TR 38.901's spatial-consistency procedure (grid + interpolation) is smooth at these scales. Diagnostic, A3 only, calibration seeds: true best-cell changes 3,541 → 1,981 per UE-hour; LOS changes 370 → 179 per UE-hour per cell. |
| 4 | **Default transmit powers** (49/37/30 dBm), with no power-calibration loop. | Phase 0 showed power barely changes failures in this interference-limited layout (0.98 % → 1.00 % for −3 dB). |
| 5 | **Tuned-A3 objective:** minimise (N_RLF + N_HOF + 0.1·N_PP,1s) per UE-hour on the calibration seeds; ties → fewer handovers. | In Phase 0 a 1-s ping-pong weighed as much as an RLF, which selected a setting with 8× more RLFs than static A3. |

## Disclosed, not tuned
- With channel v2, static A3 performs ≈ 814 handovers per UE-hour on the calibration seeds (diagnostic above).
- In the Cork traces, the operational network changed cells (LTE/5G, intra-RAT) about 40–47 times per hour while driving (one every 76–93 s).
- The simulated layout is denser than the unknown Cork layout and the speeds are 60–100 km/h, so the simulated network is a high-volatility scenario. This is reported as a limitation, and Phase 2 sweeps speed. **No further channel change is made to match the Cork rate.**

## Unchanged from Phase 0
Arms, agent and hyperparameters, constraint layer, state definition (except that s₁₁ uses the change clock), mobility (Manhattan, 60–100 km/h), training seeds (42, 43, 44) and episodes (100), evaluation scenarios (1000–1019), KPIs, statistics, and the go/no-go criterion (PREREGISTRATION.md).

## Degeneracy rule (new)
If the Reactive DQN again executes no greedy handover in evaluation, Phase 0b is reported as **degenerate**, and no further reward tuning is done without a new, separately labelled decision by the authors.

## STOP
After Phase 0b, report to the authors before Phase 1b or Phase 2.
