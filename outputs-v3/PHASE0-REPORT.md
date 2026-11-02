# Phase 0 report: go/no-go test (2026-10-27)

Protocol: `PREREGISTRATION.md`, followed without changes. Data: `data/phase0/`. Raw: `logs/raw/seed-{42,43,44}/`.

## 1. What ran
- **Calibration** (A3 and rule-only only, seeds 2000–2004). At the default power, Static A3 failed on 0.98 % of handovers, just below the 1 % floor. The pre-registered rule lowered all Tx powers by 3 dB, giving 1.00 % (accepted).
  - Tuned A3, selected on the calibration seeds: margin 5 dB, TTT 320 ms (`config/tuned-a3-grid.csv`).
  - Reward-rescaling check: 0.43 % of best-cell RSRP samples fall below −120 dBm, so no rescaling.
- **Training:** 3 learning arms × 3 seeds (42, 43, 44) × 100 episodes × 1,000 steps. 42 min per seed on CPU, 4 threads, 3 seeds in parallel.
- **Evaluation:** 20 scenarios (seeds 1000–1019), identical for all arms. About 1 min per seed.

## 2. Results (20 scenarios; learning arms averaged over the 3 seeds)

| Arm | HO / UE-h | HSR | PP (1 s) | PP (10 s) | RLF / UE-h | Outage | Mean SINR | 5th-pct SINR | Mean RSRP |
|---|---|---|---|---|---|---|---|---|---|
| Static A3 | 828 | 99.3 % | 11.3 % | 41.3 % | 4.1 | 7.7 % | 10.5 dB | −11.9 dB | −97.5 dBm |
| Tuned A3 | 475 | 99.0 % | 3.3 % | 36.0 % | 34.4 | 9.3 % | 9.8 dB | −13.5 dB | −98.0 dBm |
| Rules only | 90 | 99.8 % | 0 | 0 | 249.1 | 12.6 % | 8.4 dB | −16.3 dB | −98.9 dBm |
| Reactive DQN | **0** | — | 0 | 0 | 242.2 | 12.4 % | 8.3 dB | −15.9 dB | −98.9 dBm |
| Oracle DQN k=5 | **0** | — | 0 | 0 | 242.2 | 12.4 % | 8.3 dB | −15.9 dB | −98.9 dBm |
| Oracle DQN k=10 | **0** | — | 0 | 0 | 242.2 | 12.4 % | 8.3 dB | −15.9 dB | −98.9 dBm |

**Pre-registered verdict: NO-GO.** Both oracle arms are identical to Reactive on every decision KPI (p = 1).

## 3. The verdict is degenerate: the agents never hand over
In every seed, every learning arm learned the greedy policy "always stay": 0 handovers in all 60 evaluation runs. The ~70 handovers per training episode come only from ε = 0.05 random exploration. Foresight can't change a policy that never acts, so **this test doesn't answer whether foresight helps.** What it reveals is a defect in the reward.

**Diagnosis** (seed 42, scenario 1000, all eligible decisions, where the constraint layer would allow a handover):
- **The immediate reward barely discourages handing over.** Reactive: the median stay−HO immediate reward is +0.26, and HO is immediately better in 42 % of decisions.
- **The learned values make staying always better.** Q(stay) − Q(HO): median +27.3 (minimum +8.9) for Reactive and +46.5 for Oracle k=10. HO is preferred in **0 %** of states.
- **Cause, in three parts:**
  1. The reward has **no term for RLF, outage or handover failure**.
  2. The "long stay" term pays +0.5/+1.0/+1.5 at **every step** once 60/120/180 s have passed since the last handover, and the recency/frequency penalties all follow a handover. A handover therefore forfeits a per-step bonus for minutes.
  3. In this simulator an RLF re-attaches the UE to the strongest cell **without resetting** the time since the last handover. Failing is therefore a free handover.
- **Why v2 hid this:** in v2 RLF could never occur and RSRP degradation penalties forced the agent to move, so the problem never surfaced.

## 4. Other findings
- **Static A3 is the strongest method here** (4.1 RLF/UE-h, 7.7 % outage), at the cost of a very high handover rate (828/UE-h, one every ~4.3 s per UE) and 11 % ping-pong within 1 s.
- **That handover rate is high for vehicular traffic.** The channel is volatile: UMi shadowing decorrelates over 13 m, and the LOS state flips at 60–100 km/h. Channel realism should be checked against measured handover intervals before Phase 2 (see §5).
- **Tuned A3 is worse than Static A3 on RLF (34 vs 4 per UE-h).** The pre-registered selection objective counted every 1-s ping-pong as heavily as an RLF, so it chose a larger margin that removes ping-pongs at the cost of failures. The objective, not the grid, needs rethinking.
- **The rule-only policy behaves like never handing over** (249 RLF/UE-h). With a 30-s minimum interval and a 3-dB current-gain gate, it rarely moves at the right moment.

## 5. Proposed next step: Phase 0b, post hoc (needs approval; not run)
This changes the method, not just the environment, so it is the authors' decision.
1. **Failure-aware reward.** Add terms for what the KPIs measure:
   - a per-step outage penalty (serving SINR < Q_out);
   - an RLF penalty;
   - a handover-failure penalty.

   The other reward terms stay. The manuscript's reward table changes accordingly.
2. **Simulator semantics:** RLF re-establishment counts as a change of serving cell (it resets the time since the last handover), so failing is no longer a free handover.
3. **Channel sanity check** with baselines only: compare A3's handover interval with published drive-test values, and if needed lengthen the L3 filter or check the LOS-state correlation. Pre-registered, before any learning arm runs.
4. **Repair the tuned-A3 objective** so that an RLF weighs more than a ping-pong.
5. **Re-run the same go/no-go test** (3 arms × 3 seeds, ≈ 45 min).
