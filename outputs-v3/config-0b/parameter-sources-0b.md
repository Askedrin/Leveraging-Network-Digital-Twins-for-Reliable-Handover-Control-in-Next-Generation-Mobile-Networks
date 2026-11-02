# Phase 0b: parameter changes relative to `../config/parameter-sources.md`

All other parameters and sources are unchanged. The machine-readable values are in `pinned-config.json` (this folder).

| Parameter | Phase 0 | Phase 0b | Source / reason |
|---|---|---|---|
| Transmit powers macro / micro / pico | 46 / 34 / 27 dBm (−3 dB calibration offset) | 49 / 37 / 30 dBm (defaults, no power loop) | PREREGISTRATION-0b.md, change 4 |
| LOS-state process | first-order AR(1), d = 50 m | two cascaded AR(1) stages (Matérn-3/2), 1/e at 50 m | TR 38.901 §7.6.3 spatial consistency (grid + interpolation is smooth at step scale); change 3 |
| Shadowing process | two independent AR(1) per link (LOS, NLOS), d = 37/50 m (UMa), 10/13 m (UMi) | one Matérn-3/2 process per link, 1/e at the NLOS distance (UMa 50 m, UMi 13 m), scaled by σ of the current state | change 3 |
| Reward | v2 RSRP-based terms only | + outage −1/step, RLF −15, HOF −15 | change 1 |
| RLF re-establishment | keeps the time since last handover | resets the time since change of serving cell (reward, s₁₁) | change 2 |
| Tuned-A3 objective | N_RLF + N_HOF + N_PP,1s | N_RLF + N_HOF + 0.1·N_PP,1s | change 5 |
| Tuned-A3 selection | 5 dB, 320 ms | 6 dB, TTT ≤ 160 ms (40/80/160 ms are equivalent at 0.2 s sampling) | `tuned-a3-grid.csv` |
