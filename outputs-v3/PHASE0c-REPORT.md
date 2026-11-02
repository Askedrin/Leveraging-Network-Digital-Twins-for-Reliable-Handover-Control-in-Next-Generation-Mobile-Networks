# Phase 0c report: reliability-aware constraint layer (2026-10-28)

Protocol: `PREREGISTRATION-0c.md` (post hoc relative to 0b), followed without changes. Data: `data/phase0c/`. Raw: `logs/raw-0c/`. Config: `config-0c/` (δ_rel = 2 dB, `override-grid.csv`).

## 1. Results (scenarios 1000–1019; learning arms averaged over seeds 42–44)

| Arm | HO / UE-h | HOF rate | PP (1 s) | RLF / UE-h | Outage | 5th-pct SINR |
|---|---|---|---|---|---|---|
| Static A3 | 813 | 0.23 % | 8.1 % | 1.7 | 5.8 % | −9.3 dB |
| Tuned A3 | 620 | 0.20 % | 5.6 % | 4.1 | 6.4 % | −10.0 dB |
| **Rules only (0c layer)** | 543 | 0.31 % | 4.5 % | **0.4** | **5.4 %** | **−8.4 dB** |
| Reactive DQN | 120 | 0.41 % | 1.4 % | 213.1 | 9.3 % | −13.1 dB |
| Oracle DQN k=5 | 175 | 1.69 % | 0.0 % | 138.3 | 7.6 % | −11.3 dB |
| Oracle DQN k=10 | 98 | 3.30 % | 0.0 % | 203.8 | 8.8 % | −12.7 dB |

| Oracle vs Reactive (Holm over 4 KPIs) | k = 5 | k = 10 |
|---|---|---|
| RLF / UE-h | −35 %, p < 0.001 | −4 %, p < 0.001 |
| Outage | −19 %, p < 0.001 | −6 %, p < 0.001 |
| 5th-pct SINR | +1.8 dB (14 %), p < 0.001 | +0.4 dB (3 %), p < 0.001 |
| HOF rate | 0.41 % → 1.69 % (worse), p < 0.001 | 0.41 % → 3.30 % (worse), p < 0.001 |

**Pre-registered verdict: NO-GO.** Oracle k = 5 clears the 10 % threshold on three KPIs, but its HOF rate is significantly worse, which the criterion doesn't allow. **Degeneracy rule: not triggered** (the Reactive DQN executes greedy handovers).

## 2. What changed relative to 0b
- **The layer change works for the rule.** The rule-only policy goes from 293 to 0.4 RLF/UE-h and beats Static A3 on RLF, outage and 5th-percentile SINR, with 33 % fewer handovers and fewer ping-pongs.
- **The value of foresight is now large in relative terms** (−35 % RLF for 1 s foresight), because the agents are no longer capped at about one handover per minute.
- **The admission check reduced the oracle's HOF** from 8.1 % (0b) to 1.7 %, but didn't remove it. The check uses the *measured* (L3-filtered) SINR against Q_in. HOF is judged on the *true* SINR against Q_out at execution, and the filter lags.

## 3. The new binding factor: the reward
Every learning arm is two orders of magnitude worse than the rule-only policy that runs under the same layer (213 vs 0.4 RLF/UE-h). The agents use only 100–175 handovers per UE-hour where the channel needs about 550. The reward of the original design (Table V of the manuscript) charges a handover within 10 s of the last change −20 and within 30 s −10, plus frequency (up to −8), pattern (−25) and blacklist (−15) terms. A failure costs about −20 (outage steps + RLF). In a channel where the best cell changes every 2–7 s, following it is therefore always more expensive than failing. The failure terms added in 0b made the agents act, but not often enough.

## 4. Consequence for the campaign
PREREGISTRATION-2 fixed the campaign before these results, with the original reward. Run as registered, its outcome on H3/H4 is predictable from this table: every DQN arm far worse than the rule and A3. That would test the reward of the original design, not the coupling. The campaign is therefore amended **before any Phase 2 run** (`PREREGISTRATION-2-AMENDMENT.md`, labelled post hoc relative to 0c):
- the learning arms use a KPI-aligned reward, developed on training and calibration scenarios only;
- the Phase 0c agents (original reward) are evaluated on the fresh campaign scenarios as a reported reference.
