#!/usr/bin/env python3
"""P13 (layout) and P18 (twin validation CDFs), PREREGISTRATION-2.md section 5.

Policy-free statistics on the sweep scenarios 3000-3019 of the campaign configuration:
coupling loss (path loss + shadowing - BS antenna gain) to the best-RSRP cell, the best-cell
RSRP per RE, and the wideband SINR of that cell. Also the true best-cell change rate.
Writes ../data/layout/cells.csv, ../data/twin-validation/{cdf.csv,stats.json}.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from campaign import SWEEP, base_config
from twin import OUTPUT_ROOT, Scenario, cell_kind_index


def main():
    cfg, _ = base_config()
    lay = OUTPUT_ROOT / "data" / "layout"
    val = OUTPUT_ROOT / "data" / "twin-validation"
    lay.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)
    cl, rsrp, sinr, changes, ueh = [], [], [], 0, 0.0
    for s in SWEEP:
        sc = Scenario(cfg, s)
        n = cfg.steps_per_episode
        kind = np.array([cell_kind_index(c) for c in sc.cells])
        p_tx = np.array(cfg.tx_power_dbm)[kind] + cfg.tx_offset_db
        re_db = 10 * math.log10(12 * cfg.n_rb)
        truth = sc.truth[:n]
        best = truth.argmax(axis=2)
        t_i, v_i = np.meshgrid(np.arange(n), np.arange(cfg.n_vehicles), indexing="ij")
        r_best = truth[t_i, v_i, best]
        cl.append((p_tx[best] - re_db - r_best).ravel())
        rsrp.append(r_best.ravel())
        sinr.append(sc.sinr_true[:n][t_i, v_i, best].ravel())
        changes += int((np.diff(best, axis=0) != 0).sum())
        ueh += n * cfg.n_vehicles * cfg.dt_s / 3600
        if s == SWEEP[0]:
            cells = pd.DataFrame([{"cell": i, "type": c.cell_type, "x_m": c.x_m, "y_m": c.y_m,
                                   "tx_power_dbm": float(p_tx[i]),
                                   "antenna_gain_dbi": cfg.antenna_gain_dbi[kind[i]],
                                   "height_m": cfg.h_bs_m[kind[i]],
                                   "scenario": "UMi-Street Canyon" if kind[i] else "UMa"}
                                  for i, c in enumerate(sc.cells)])
            cells.to_csv(lay / "cells.csv", index=False)
            traj = pd.DataFrame({"t": np.repeat(np.arange(n), 3),
                                 "ue": np.tile(np.arange(3), n),
                                 "x_m": sc.x[:n, :3].ravel(), "y_m": sc.y[:n, :3].ravel()})
            traj.to_csv(lay / "example-trajectories.csv", index=False)
    rows = []
    q = np.linspace(0, 1, 201)
    for name, arr in (("coupling_loss_db", cl), ("best_rsrp_dbm_per_re", rsrp),
                      ("best_cell_sinr_db", sinr)):
        a = np.concatenate(arr)
        rows += [{"metric": name, "cdf": float(p), "value": float(v)}
                 for p, v in zip(q, np.quantile(a, q))]
    pd.DataFrame(rows).to_csv(val / "cdf.csv", index=False)
    stats = {}
    for name, arr in (("coupling_loss_db", cl), ("best_rsrp_dbm_per_re", rsrp),
                      ("best_cell_sinr_db", sinr)):
        a = np.concatenate(arr)
        stats[name] = {p: float(np.percentile(a, int(p[1:]))) for p in ("p5", "p50", "p95")}
    stats["true_best_cell_changes_per_ueh"] = changes / ueh
    stats["frac_sinr_below_q_out"] = float((np.concatenate(sinr) < cfg.q_out_db).mean())
    stats["scenarios"] = [SWEEP[0], SWEEP[-1]]
    (val / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
