#!/usr/bin/env python3
"""Development of c_HO for the KPI-aligned reward (PREREGISTRATION-2-AMENDMENT.md).

  python reward_dev.py --ho-cost 0.1        one pilot: Reactive DQN, seed 41, 40 episodes,
                                            greedy evaluation on calibration scenarios 2000-2004
  python reward_dev.py --select             pick c_HO, write ../config-2/reward-dev.csv
Training and calibration scenarios only; no evaluation scenario is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import pandas as pd
import torch

from run_seed import load_config
from twin import OUTPUT_ROOT, DQNAgent, Runtime, Scenario

SEED, EPISODES, CANDIDATES = 41, 40, (0.0, 0.1)
DEV = OUTPUT_ROOT / "logs" / "reward-dev"
CFG2 = OUTPUT_ROOT / "config-2"


def pilot(ho_cost: float, threads: int):
    torch.set_num_threads(threads)
    cfg, _ = load_config("config-0c")
    cfg = replace(cfg, reward_mode="kpi", ho_cost=ho_cost)
    agent = DQNAgent(cfg, SEED, torch.device("cpu"))
    for ep in range(EPISODES):
        rt = Runtime(cfg, Scenario(cfg, SEED * 1_000_000 + ep))
        for t in range(cfg.steps_per_episode):
            rt.step_agent(agent, t, 0, training=True)
            agent.train_update()
        m = rt.metrics("pilot", SEED, ep)
        print(f"c_ho={ho_cost} ep {ep + 1}/{EPISODES} ho={m['n_ho']} rlf={m['n_rlf']} "
              f"out={m['outage_frac']:.3f}", flush=True)
    rows = []
    for s in cfg.calibration_seeds:
        rt = Runtime(cfg, Scenario(cfg, s))
        for t in range(cfg.steps_per_episode):
            rt.step_agent(agent, t, 0, training=False)
        rows.append(rt.metrics("Reactive DQN (pilot)", SEED, s))
    df = pd.DataFrame(rows)
    ueh = df.ue_steps.sum() * cfg.dt_s / 3600
    res = {"ho_cost": ho_cost, "ho_per_ueh": df.n_ho.sum() / ueh,
           "rlf_per_ueh": df.n_rlf.sum() / ueh, "hof_rate": df.n_hof.sum() / max(df.n_ho.sum(), 1),
           "ppr1": df.n_pp1.sum() / max(df.n_ho.sum(), 1),
           "outage_frac": df.outage_steps.sum() / df.ue_steps.sum(),
           "objective_per_ueh": (df.n_rlf.sum() + df.n_hof.sum() + 0.1 * df.n_pp1.sum()) / ueh}
    DEV.mkdir(parents=True, exist_ok=True)
    (DEV / f"pilot-{ho_cost:g}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)


def select():
    rows = [json.loads((DEV / f"pilot-{c:g}.json").read_text()) for c in CANDIDATES]
    df = pd.DataFrame(rows)
    df["greedy_handovers"] = df.ho_per_ueh > 0
    ok = df[df.greedy_handovers].sort_values(["objective_per_ueh", "ho_per_ueh"])
    CFG2.mkdir(parents=True, exist_ok=True)
    df.to_csv(CFG2 / "reward-dev.csv", index=False)
    if ok.empty:
        raise SystemExit("amendment fails: no candidate executes greedy handovers")
    best = float(ok.iloc[0].ho_cost)
    (CFG2 / "reward.json").write_text(json.dumps(
        {"reward_mode": "kpi", "ho_cost": best, "pp_penalty": 1.5, "candidates": CANDIDATES,
         "rule": "min (N_RLF + N_HOF + 0.1 N_PP,1s)/UE-h of the pilot on calibration seeds"},
        indent=2) + "\n")
    print(df.to_string(index=False))
    print("selected c_HO =", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ho-cost", type=float)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    select() if a.select else pilot(a.ho_cost, a.threads)
