# v3 simulated network — parameter sources

"Design choice" = chosen by us, stated as such in the paper; "VERIFY" = value recalled from the
cited document; check the exact table before submission.

| Parameter | Value | Source |
|---|---|---|
| Area, layout | 2000 × 1500 m; 7 macro (centre + 6 on a 600 m ring), 4 micro, 1 pico | v2 `make_cells` (unchanged); design choice |
| Carrier frequency | 3.5 GHz | Design choice (NR band n78) |
| Bandwidth, SCS, N_RB | 100 MHz, 30 kHz, 273 PRB | TS 38.101-1 Table 5.3.2-1 (100 MHz @ 30 kHz → 273 PRB) — VERIFY |
| Tx power macro / micro / pico | 49 / 37 / 30 dBm, then −3 dB calibration offset → 46 / 34 / 27 dBm | Design choice; offset set by the pre-registered calibration (`calibration.json`) |
| Antenna gain macro / micro / pico, UE | 8 / 5 / 5 dBi, 0 dBi; omnidirectional | Design choice (simplification; no sectorisation) |
| BS height macro / micro, pico; UE height | 25 m / 10 m; 1.5 m | TR 38.901 UMa / UMi-SC scenario defaults |
| Path loss | UMa (macro), UMi-Street Canyon (micro, pico), LOS and NLOS formulas, breakpoint with h_E = 1 m | TR 38.901 Table 7.4.1-1 |
| LOS probability | UMa / UMi formulas for h_UT ≤ 13 m | TR 38.901 Table 7.4.2-1 |
| LOS-state spatial consistency | correlated Gaussian (AR(1) in travelled distance) mapped through Φ, compared with Pr_LOS; correlation distance 50 m (UMa, UMi) | TR 38.901 §7.6.3.3 procedure; distance — VERIFY (Table 7.6.3.1-2) |
| Shadow fading σ (LOS / NLOS) | UMa 4 / 6 dB; UMi 4 / 7.82 dB | TR 38.901 Table 7.4.1-1 — VERIFY |
| Shadowing decorrelation distance (LOS / NLOS) | UMa 37 / 50 m; UMi 10 / 13 m | TR 38.901 Table 7.5-6 Part-1 — VERIFY |
| Shadowing correlation model | AR(1) in travelled distance, ρ = exp(−Δd/d_corr) | Gudmundson 1991 |
| RSRP | per resource element: P_tx − 10log10(12·N_RB) + G − PL − SF | Definition of RSRP (TS 38.215) — VERIFY clause |
| Residual measurement fading | N(0, 1 dB) i.i.d. per 0.2 s sample | Design choice (sensitivity 2, 4 dB planned) |
| L3 filter | F_n = (1−a)F_{n−1} + a·M_n, a = 2^(−k/4), k = 4 | TS 38.331 §5.5.3.2 |
| Noise | −174 dBm/Hz + 10log10(30 kHz) + NF 9 dB = −120.2 dBm per RE | NF 9 dB: common 3GPP UE assumption (TR 38.901 §7.8 calibration) — VERIFY |
| Cell load → interference factor | load ∈ {0..30} (v2 process), η = 0.2 + 0.6·load/30 | Design choice |
| Q_out / Q_in | −8 dB / −6 dB SINR | Commonly used mapping of TS 38.133 RLM (10 % / 2 % hypothetical PDCCH BLER) — VERIFY clause |
| T310 | 1 s (5 steps); timer stops when SINR > Q_in | TS 38.331 timer (value from allowed range); design choice |
| T_store (MRO classification) | 5 s | Design choice (TS 38.300 MRO concept) |
| Ping-pong minimum time of stay | 1 s (primary), 10 s (secondary) | TR 36.839 (MTS = 1 s) — VERIFY |
| Measurement-report threshold r_min | −125 dBm | Design choice |
| Mobility | Manhattan grid, 250 m blocks, straight 0.5 / left 0.25 / right 0.25 | Design choice (standard Manhattan model) |
| Speeds | 60–100 km/h, ±2 km/h per step | Design choice (urban arterial) |
| Vehicles, step, episode | 20, 0.2 s, 1,000 steps | v2 (unchanged) |
| Agent, reward, constraint layer | unchanged | `ch5-outputs-v2/scripts/common.py` (imported) |
