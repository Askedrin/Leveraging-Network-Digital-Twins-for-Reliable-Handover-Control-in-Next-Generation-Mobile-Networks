# Amendment to PREREGISTRATION-2.md: KPI-aligned reward

**Written:** 2026-10-28, after Phase 0c (`PHASE0c-REPORT.md`) and before any Phase 2 learning-arm training or evaluation. Not edited after Phase 2 learning-arm results are seen. **Labelled post hoc** relative to Phase 0c.

**Disclosure.** The seed-independent baselines (Static, Tuned and MRO A3, both rule-only variants), with parameters fixed in `config-0c/` and `config-2/`, had already been run on the campaign scenarios (`logs/raw-2/baselines/`), and their results were seen. No baseline parameter is changed by this amendment.

## Why
In Phase 0c every learning arm was two orders of magnitude worse than the rule-only policy under the same constraint layer (213 vs 0.4 RLF/UE-h). The cause is the original design's handover-penalty shaping (PHASE0c-REPORT.md §3). Run with that reward, the campaign would evaluate the reward, not the coupling of forecaster and agent.

## Change (all learning arms; everything else in PREREGISTRATION-2.md is unchanged)
The shaped reward of the original design is replaced by a reward made of the evaluation KPIs. For each UE and step:

R_t = − 1[serving SINR < Q_out] − 15·1[RLF] − 15·1[HOF] − c_HO·1[handover executed] − 1.5·1[1-s ping-pong]

| Term | Rule |
|---|---|
| Outage and RLF | Credited to the previous action, as in Phase 0b. |
| HOF, handover cost and ping-pong | Credited to the executing action. A ping-pong is a handover back to the previous cell within T_MTS = 1 s. |
| Weights | RLF : HOF : ping-pong = 1 : 1 : 0.1, the objective already used to tune A3, MRO and δ_rel, so that every tuned method optimises the same objective. The outage term gives the dense signal that precedes an RLF. |

## Development of c_HO (training and calibration data only)
- **Candidates:** c_HO ∈ {0, 0.1}.
- **Pilot runs:** Reactive DQN only, training seed 41 (not a campaign seed), 40 episodes on its training scenarios (41·10⁶ + episode), Phase 0c layer.
- **Selection:**
  - Both candidates are evaluated greedily on the calibration scenarios 2000–2004.
  - The lower (N_RLF + N_HOF + 0.1·N_PP,1s) per UE-hour wins; ties go to fewer handovers.
  - If neither candidate executes any greedy handover, the amendment fails. It is then reported, and the campaign runs as originally registered.
- **Records:** `config-2/reward-dev.csv` and `config-2/pinned-config.json`.

## Additional reference arms (reported, not in the hypotheses)
- Phase 0c agents (seeds 42–44, original reward, 0c layer), Reactive and Oracle k = 5, on the headline scenarios: "original reward".
- Phase 0b agents (paper's layer, original reward), as already registered.

## Unchanged
- Arms, seeds 42–46 and 100 episodes.
- Forecaster selection rule and τ_e.
- Scenarios, KPIs, statistics, hypotheses H1–H5 and sweeps.
