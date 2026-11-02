# Phase 0b report: go/no-go re-test (2026-10-27)

Protocol: `PREREGISTRATION-0b.md` (post hoc relative to Phase 0), followed without changes. Data: `data/phase0b/`. Raw: `logs/raw-0b/seed-{42,43,44}/`. Config: `config-0b/`.

## 1. What ran
- **Calibration** (baselines only, seeds 2000–2004): default Tx powers, no power loop. Tuned A3 selected with the repaired objective (N_RLF + N_HOF + 0.1·N_PP,1s): margin 6 dB, TTT ≤ 160 ms (`config-0b/tuned-a3-grid.csv`).
- **Training:** 3 learning arms × 3 seeds × 100 episodes × 1,000 steps, with the failure-aware reward, RLF re-establishment counted as a change of serving cell, and channel v2. About 52 min per seed (CPU, 4 threads, 3 seeds in parallel).
- **Evaluation:** 20 scenarios (1000–1019), identical for all arms. About 1 min per seed.

## 2. Results (20 scenarios; learning arms averaged over the 3 seeds)

| Arm | HO / UE-h | HOF rate | PP (1 s) | PP (10 s) | RLF / UE-h | Outage | Mean SINR | 5th-pct SINR |
|---|---|---|---|---|---|---|---|---|
| Static A3 | 813 | 0.23 % | 8.1 % | 37.8 % | **1.7** | **5.8 %** | 11.7 dB | **−9.3 dB** |
| Tuned A3 (6 dB) | 620 | 0.20 % | 5.6 % | 33.7 % | 4.1 | 6.4 % | 11.3 dB | −10.0 dB |
| Rules only | 90 | 0 | 0 | 0 | 292.8 | 10.9 % | 9.4 dB | −14.7 dB |
| Reactive DQN | 67 | 0.36 % | 0 | 0 | 250.9 | 10.0 % | 9.6 dB | −13.7 dB |
| Oracle DQN k=5 (1 s) | 67 | 8.1 % | 0 | 0 | 243.3 | 9.6 % | 9.7 dB | −13.5 dB |
| Oracle DQN k=10 (2 s) | 52 | 17.5 % | 0 | 0 | 250.5 | 9.8 % | 9.7 dB | −13.6 dB |

Paired Wilcoxon, oracle vs reactive, Holm over the 4 decision KPIs (`paired_tests.csv`):

| KPI | Oracle k=5 vs Reactive | Oracle k=10 vs Reactive |
|---|---|---|
| RLF / UE-h | −3.0 %, p_Holm < 0.001 | −0.1 %, n.s. |
| Outage fraction | −3.9 %, p_Holm < 0.001 | −1.8 %, p_Holm < 0.001 |
| 5th-pct SINR | +0.17 dB (1.3 %), p_Holm < 0.001 | +0.08 dB, n.s. |
| HOF rate | **0.36 % → 8.1 %** (worse), p_Holm < 0.001 | **0.36 % → 17.5 %** (worse), p_Holm < 0.001 |

**Pre-registered verdict: NO-GO.** Perfect 1-s foresight gives statistically significant but small gains (3–4 %), below the 10 % threshold. HOF also degrades significantly. Perfect 2-s foresight is no better than reactive on RLF.

**Degeneracy rule: not triggered.** The Reactive DQN executes greedy handovers in evaluation (67 per UE-hour, all agent-chosen and passed by the gate). Unlike Phase 0, this NO-GO is a real answer to the question asked.

## 3. Why the gains are small: the constraint layer is the binding factor
**Diagnostic, not pre-registered and not part of the criterion.** Script: `scripts/diag_constraint_layer.py`; data: `data/phase0b/diag-constraint-layer.csv`. Rule-only policy on the same 20 scenarios, switching off one part of the constraint layer at a time. No training.

| Rule-only variant | HO / UE-h | PP (1 s) | RLF / UE-h | Outage | 5th-pct SINR |
|---|---|---|---|---|---|
| Full layer (30 s interval, blacklist, 3 dB) | 90 | 0 | 292.8 | 10.9 % | −14.7 dB |
| Interval 0 s (blacklist kept) | 83 | 9.0 % | 274.0 | 10.4 % | −14.1 dB |
| No blacklist (30 s interval kept) | 108 | 0 | 293.1 | 10.9 % | −14.8 dB |
| No blacklist, interval 1 s | 887 | 0 | 31.5 | 5.4 % | −8.8 dB |
| No blacklist, interval 0 s (3-dB gain only) | 987 | 13.3 % | **0.0** | **3.6 %** | **−5.5 dB** |
| No blacklist, interval 0 s, gain 0 dB | 1487 | 25.7 % | 0.0 | 3.1 % | −3.9 dB |

- **The failures come from the anti-ping-pong limits, not from the decision policy.** The blacklist (more than 4 HOs in 300 s or more than 2 in 60 s blocks the UE for 120 s) and the 30-s minimum interval each cap the handover rate near 100 per UE-hour on their own. This channel needs about 800–1,000 per UE-hour to follow the best cell. With both limits removed, the same greedy 3-dB rule has **0 RLF** and less outage than Static A3.
- **What this means for the DQN arms:** the layer allows at most one handover per 30 s, and the learning arms use about one per minute (52–67 per UE-hour). Foresight can only help choose *which* of those rare opportunities to use, which bounds its value from the start. The Phase 0b result answers "does foresight help an agent inside this constraint layer?" (barely). It does not answer "does foresight help handover control?"
- **Why foresight raises HOF:** the execution gate uses the arm's own signal source (`twin.py`, `gate`). The oracle therefore hands over to a cell that *will* be ≥ 3 dB better in 1–2 s, but whose SINR *now* may still be below Q_out. HOF is evaluated at execution time, so those handovers fail.
- **This is scenario-dependent.** The simulated channel is high-volatility: Static A3 performs ≈ 813 HO/UE-h, against 40–47 cell changes per hour in the Cork drive traces (disclosed in PREREGISTRATION-0b.md). At the Cork rate, the same limits (about 100/UE-h) would rarely bind. The finding "the constraint layer causes RLFs" holds for this scenario, not in general.

## 4. Other findings
- **Static A3 remains the strongest method on reliability** (1.7 RLF/UE-h, 5.8 % outage), at 813 HO/UE-h and 8 % 1-s ping-pong.
- **The repaired Tuned-A3 objective works as intended:** 24 % fewer handovers and fewer ping-pongs than Static A3, at a moderate RLF cost (4.1 vs 1.7). In Phase 0 the cost was 34 vs 4.
- **The failure-aware reward did what it was designed to do:** the agents now hand over (67 HO/UE-h, against 0 in Phase 0). In the same channel, the Reactive DQN beats the rule-only policy on RLF (251 vs 293, −14 %) and on outage (10.0 % vs 10.9 %), with fewer handovers. Phase 0 and 0b RLF values are not comparable because the channel changed (v1 → v2).
- **Training converged:** mean episode reward rose from about −0.3 (episodes 1–10) to about 1.5 (episodes 50–100) for every arm and seed, and was flat after episode 50.

## 5. Implications for the manuscript
- The original claims (large reliability gains from proactive DQN + constraint layer) are **not supported** in a twin whose ground truth has no LSTM in it.
- Two results are solid and reportable:
  1. The oracle-bounded value of foresight inside the constraint layer is small (≤ 4 %), and it comes with a HOF cost unless the gate checks the *current* target SINR.
  2. Anti-ping-pong limits trade reliability for stability. In a high-volatility channel this trade dominates every learning effect.
- The constraint layer as described in the manuscript (30 s, blacklist) must not be presented as a reliability mechanism. In this scenario it is the main cause of failures.

## 6. Options (authors' decision; nothing run)
Per PREREGISTRATION-0b.md, stopped here. Each option below is a new, separately labelled decision.

- **A. Stop experimenting; reframe (plan §4, Option B).** Write an honest decomposition paper: NDT-hosted handover function, oracle-bounded value of foresight, and the constraint-layer reliability trade-off (this report §2–3), plus Phase 1a (real-trace forecasters vs persistence). Weakest novelty, but the strongest evidence. Workshop or short-conference scale.
- **B. Phase 0c: reliability-aware constraint layer (post hoc, one change).**
  - When the UE's *measured* serving SINR is near Q_out (margin chosen on the calibration seeds), the minimum interval and blacklist are bypassed for that decision.
  - The gate also requires the target's *current* measured SINR ≥ Q_in, which removes the oracle's HOF mechanism.
  - Everything else from 0b stays; re-run the same go/no-go test, ≈ 1 h compute.
  - This is the "reliable" in the title and a genuine coupling mechanism. It is a third post-hoc iteration, though, and must be disclosed as such.
- **C. Phase 0c': scenario at the Cork volatility.** Recalibrate the scenario (speeds, layout density) so that Static A3's HO rate matches the Cork traces, as a pre-set target *before* any learning arm runs. Then re-run the go/no-go test. This addresses the realism caveat, but changing the environment after a NO-GO looks like a search for a favourable setting. Reviewers will discount it unless it is framed as a pre-registered scenario sweep (originally planned in Phase 2).

**Recommendation: B.**
1. It changes the method, not the environment, and the change is motivated by a diagnosed mechanism (§3), not by the KPI outcome.
2. It is also what the paper's "reliable" claim needs.
3. If B is still a NO-GO, fall back to A with B's result included.
4. C belongs in Phase 2 as a pre-registered speed and density sweep, whichever option is chosen.
