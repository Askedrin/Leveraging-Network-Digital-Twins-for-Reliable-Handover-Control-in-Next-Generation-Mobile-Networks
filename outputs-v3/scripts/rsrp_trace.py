#!/usr/bin/env python3
"""Serving-cell RSRP over time (the original paper's RSRP-evolution figure, Study 3).

Static A3 and the proposed agent (Proactive DQN + fallback, 5 seeds) on the 50 headline
scenarios. At every step the true RSRP of each UE's serving cell is recorded where the
runtime records it for the mean_rsrp_dbm metric (observe). Writes, per part,
../data/rsrp-trace-3/part-<i>.csv with one row per (arm, seed, scenario, step).

    python rsrp_trace.py --part i --parts n      # then: python rsrp_trace.py --merge
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

import campaign
from campaign import (ARM_K, ARMS, A3Controller, CampaignRuntime, base_config, configure_study,
                      forecast_tools)
from twin import OUTPUT_ROOT, DQNAgent, Runtime, Scenario

OUT = OUTPUT_ROOT / "data" / "rsrp-trace-3"
SEEDS = [42, 43, 44, 45, 46]
ARM = "Proactive DQN + fallback"


def logged(rt, sc, log):
    """Record the mean serving RSRP of all UEs each time the runtime observes step t."""
    observe, idx = rt.observe, np.arange(rt.nv)

    def wrapped(t):
        log.append((t, float(sc.truth[t, idx, rt.serving].mean()),
                    float(sc.truth[t, idx, rt.serving].min())))
        observe(t)
    rt.observe = wrapped


def run(part, parts, threads):
    torch.set_num_threads(threads)
    cfg, baselines = base_config()
    fcm, tau, _ = forecast_tools()
    a3s = baselines["static_a3"]
    agents = {}
    for s in SEEDS:
        agents[s] = DQNAgent(cfg, s, torch.device("cpu"))
        agents[s].load(campaign.RAW / f"seed-{s}" / f"agent-{ARMS[ARM]}.pth")
    rows = []
    for scen in campaign.HEADLINE[part::parts]:
        sc = Scenario(cfg, scen)
        fc = fcm.predict(sc)
        runs = [("Static A3", "n/a", None)] + [(ARM, s, agents[s]) for s in SEEDS]
        for arm, seed, agent in runs:
            log = []
            if agent is None:
                rt, ctl = Runtime(cfg, sc), A3Controller(cfg, a3s["margin_db"], a3s["ttt_s"])
                logged(rt, sc, log)
                for t in range(cfg.steps_per_episode):
                    ctl.step(rt, t)
            else:
                rt = CampaignRuntime(cfg, sc, ARMS[arm], fc, tau)
                logged(rt, sc, log)
                for t in range(cfg.steps_per_episode):
                    rt.step_agent(agent, t, ARM_K[ARMS[arm]], training=False)
            rows += [{"arm": arm, "training_seed": seed, "scenario": scen, "step": t,
                      "time_s": t * cfg.dt_s, "mean_rsrp_dbm": m, "min_rsrp_dbm": lo}
                     for t, m, lo in log]
        print("scenario", scen, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"part-{part}.csv", index=False)


def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("part-*.csv"))], ignore_index=True)
    df.to_csv(OUT / "per_step.csv.gz", index=False)
    # Learning arm: average the seeds first, as in the campaign's per-scenario metrics.
    per = (df.groupby(["arm", "scenario", "time_s"]).mean_rsrp_dbm.mean().reset_index())
    out = per.groupby(["arm", "time_s"]).mean_rsrp_dbm.agg(
        mean="mean", p5=lambda x: np.percentile(x, 5), p95=lambda x: np.percentile(x, 95))
    out.reset_index().to_csv(OUT / "rsrp_vs_time.csv", index=False)
    print(per.groupby("arm").mean_rsrp_dbm.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    configure_study("3")
    if a.merge:
        merge()
    else:
        run(a.part, a.parts, a.threads)


if __name__ == "__main__":
    main()
