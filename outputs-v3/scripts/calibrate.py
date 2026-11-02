#!/usr/bin/env python3
"""Pre-registered scenario calibration and tuned-A3 selection (A3 and Rules only).

1. Transmit-power offset: Static A3 must show (N_HOF + N_RLF) / N_HO in [1 %, 10 %]
   over the calibration seeds; otherwise shift all Tx powers by 3 dB (range -9..+6 dB).
2. Tuned A3: grid over margin x TTT on the calibration seeds; minimise
   (N_RLF + N_HOF + N_PP,1s) per UE-hour, ties -> fewer handovers.
No learning arm is run here.  Writes ../config/{calibration,baselines,pinned-config}.json.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from twin import (ARM_A3, ARM_RULES, OUTPUT_ROOT, A3Controller, Runtime, Scenario, TwinConfig,
                  runtime_hardware)

STATIC_A3 = (3.0, 0.16)
MARGINS = [0, 1, 2, 3, 4, 5, 6]
TTTS = [0.04, 0.08, 0.16, 0.32, 0.64]


def run_a3(cfg, sc, margin, ttt, seed):
    rt, a3 = Runtime(cfg, sc), A3Controller(cfg, margin, ttt)
    for t in range(cfg.steps_per_episode):
        a3.step(rt, t)
    return rt.metrics(ARM_A3, "n/a", seed)


def run_rules(cfg, sc, seed):
    rt = Runtime(cfg, sc)
    for t in range(cfg.steps_per_episode):
        rt.step_rules(t)
    return rt.metrics(ARM_RULES, "n/a", seed)


def failure_ratio(rows):
    df = pd.DataFrame(rows)
    return (df.n_hof.sum() + df.n_rlf.sum()) / max(df.n_ho.sum(), 1)


# Phase 0b (post hoc, PREREGISTRATION-0b.md): channel v2, failure-aware reward,
# re-establishment counts as a change of serving cell, default powers (no power
# calibration), tuned-A3 objective weighs a 1-s ping-pong at 0.1 of a failure.
PHASES = {
    "0": dict(base={}, power_loop=True, pp_weight=1.0, out="config"),
    "0b": dict(base=dict(channel_version=2, failure_aware_reward=True,
                         reestablish_is_change=True), power_loop=False, pp_weight=0.1,
               out="config-0b"),
    # Phase 0c (post hoc, PREREGISTRATION-0c.md): 0b + reliability-aware constraint layer;
    # the override margin is selected with the Rules-only policy on the calibration seeds
    "0c": dict(base=dict(channel_version=2, failure_aware_reward=True,
                         reestablish_is_change=True, reliability_override=True,
                         admission_check=True), power_loop=False, pp_weight=0.1,
               out="config-0c", override_grid=[0.0, 2.0, 4.0, 6.0, 8.0]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="0")
    ph = PHASES[ap.parse_args().phase]
    out = OUTPUT_ROOT / ph["out"]
    out.mkdir(parents=True, exist_ok=True)
    log, offset, tried = [], 0.0, set()
    t0 = time.perf_counter()
    while True:
        cfg = replace(TwinConfig(), tx_offset_db=offset, **ph["base"])
        scen = {s: Scenario(cfg, s) for s in cfg.calibration_seeds}
        a3 = [run_a3(cfg, scen[s], *STATIC_A3, s) for s in cfg.calibration_seeds]
        rules = [run_rules(cfg, scen[s], s) for s in cfg.calibration_seeds]
        ratio = failure_ratio(a3)
        below = float(np.mean([(sc.truth[:cfg.steps_per_episode].max(axis=2) < -120).mean()
                               for sc in scen.values()]))
        entry = {"tx_offset_db": offset, "a3_failure_ratio": ratio,
                 "a3_totals": pd.DataFrame(a3)[["n_ho", "n_hof", "n_rlf", "n_pp1"]].sum().to_dict(),
                 "a3_means": pd.DataFrame(a3)[["outage_frac", "p5_sinr_db",
                                               "mean_rsrp_dbm"]].mean().to_dict(),
                 "rules_totals": pd.DataFrame(rules)[["n_ho", "n_hof", "n_rlf",
                                                      "n_pp1"]].sum().to_dict(),
                 "rules_means": pd.DataFrame(rules)[["outage_frac", "p5_sinr_db",
                                                     "mean_rsrp_dbm"]].mean().to_dict(),
                 "frac_best_cell_below_-120dBm": below}
        log.append(entry)
        print(json.dumps(entry, default=float), flush=True)
        tried.add(offset)
        if not ph["power_loop"]:
            break
        if ratio < 0.01 and offset - 3 >= -9 and offset - 3 not in tried:
            offset -= 3
        elif ratio > 0.10 and offset + 3 <= 6 and offset + 3 not in tried:
            offset += 3
        else:
            break
    accepted = 0.01 <= log[-1]["a3_failure_ratio"] <= 0.10
    cfg = replace(TwinConfig(), tx_offset_db=offset, **ph["base"])
    scen = {s: Scenario(cfg, s) for s in cfg.calibration_seeds}

    if "override_grid" in ph:
        ogrid = []
        for margin in ph["override_grid"]:
            c = replace(cfg, override_margin_db=margin)
            df = pd.DataFrame([run_rules(c, scen[s], s) for s in cfg.calibration_seeds])
            ueh = df.ue_steps.sum() * cfg.dt_s / 3600
            ogrid.append({"override_margin_db": margin,
                          "objective_per_ueh": (df.n_rlf.sum() + df.n_hof.sum() +
                                                ph["pp_weight"] * df.n_pp1.sum()) / ueh,
                          "ho_per_ueh": df.n_ho.sum() / ueh,
                          "rlf_per_ueh": df.n_rlf.sum() / ueh,
                          "hof_rate": df.n_hof.sum() / max(df.n_ho.sum(), 1),
                          "ppr1": df.n_pp1.sum() / max(df.n_ho.sum(), 1),
                          "outage_frac": df.outage_steps.sum() / df.ue_steps.sum()})
            print(json.dumps(ogrid[-1], default=float), flush=True)
        ogrid_df = pd.DataFrame(ogrid).sort_values(["objective_per_ueh", "ho_per_ueh"])
        ogrid_df.to_csv(out / "override-grid.csv", index=False)
        cfg = replace(cfg, override_margin_db=float(ogrid_df.iloc[0].override_margin_db))

    grid = []
    for margin in MARGINS:
        for ttt in TTTS:
            rows = [run_a3(cfg, scen[s], margin, ttt, s) for s in cfg.calibration_seeds]
            df = pd.DataFrame(rows)
            ueh = df.ue_steps.sum() * cfg.dt_s / 3600
            grid.append({"margin_db": margin, "ttt_s": ttt,
                         "objective_per_ueh": (df.n_rlf.sum() + df.n_hof.sum() +
                                               ph["pp_weight"] * df.n_pp1.sum()) / ueh,
                         "ho_per_ueh": df.n_ho.sum() / ueh})
    grid_df = pd.DataFrame(grid).sort_values(["objective_per_ueh", "ho_per_ueh"])
    grid_df.to_csv(out / "tuned-a3-grid.csv", index=False)
    best = grid_df.iloc[0]
    baselines = {"static_a3": {"margin_db": STATIC_A3[0], "ttt_s": STATIC_A3[1],
                               "note": "offset 0 dB + hysteresis 3 dB (v2); fires on the 2nd "
                                       "consecutive 0.2 s sample when TTT <= 0.2 s"},
                 "tuned_a3": {"margin_db": float(best.margin_db), "ttt_s": float(best.ttt_s),
                              "selection": f"min (N_RLF + N_HOF + {ph['pp_weight']} N_PP,1s) "
                                           "per UE-hour on calibration seeds, ties -> fewer HOs"}}
    (out / "baselines.json").write_text(json.dumps(baselines, indent=2) + "\n")
    (out / "calibration.json").write_text(json.dumps(
        {"rule": ("Static A3 (N_HOF+N_RLF)/N_HO in [0.01, 0.10]; Tx offset steps of 3 dB "
                  "in [-9, +6]") if ph["power_loop"] else "Phase 0b: default powers, no loop",
         "log": log, "final_tx_offset_db": offset,
         "accepted": accepted, "runtime_s": time.perf_counter() - t0}, indent=2, default=float)
        + "\n")
    pinned = asdict(cfg)
    pinned["hardware"] = runtime_hardware()
    (out / "pinned-config.json").write_text(json.dumps(pinned, indent=2, default=str) + "\n")
    print("final offset", offset, "accepted", accepted, "tuned A3", baselines["tuned_a3"])


if __name__ == "__main__":
    main()
