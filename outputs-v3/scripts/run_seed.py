#!/usr/bin/env python3
"""Phase 0: train the three learning arms for one training seed, then evaluate every arm.

Usage: python run_seed.py --seed 42 [--threads 4] [--episodes 100] [--steps 1000]
Writes ../logs/raw/seed-<seed>/. Aggregation and the go/no-go test: aggregate.py.
Requires ../config/{pinned-config,baselines}.json from calibrate.py.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields, replace

import pandas as pd
import torch

from twin import (ARM_A3, ARM_A3T, ARM_RULES, LEARNING_ARMS, OUTPUT_ROOT, A3Controller, DQNAgent,
                  Runtime, Scenario, TwinConfig)


def load_config(config_dir: str = "config") -> tuple[TwinConfig, dict]:
    pinned = json.loads((OUTPUT_ROOT / config_dir / "pinned-config.json").read_text())
    names = {f.name for f in fields(TwinConfig)}
    kwargs = {k: (tuple(tuple(x) if isinstance(x, list) else x for x in v)
                  if isinstance(v, list) else v)
              for k, v in pinned.items() if k in names}
    baselines = json.loads((OUTPUT_ROOT / config_dir / "baselines.json").read_text())
    return TwinConfig(**kwargs), baselines


def train(cfg, seed, device, out, episodes):
    agents = {arm: DQNAgent(cfg, seed, device) for arm in LEARNING_ARMS}
    rows = []
    start = time.perf_counter()
    for ep in range(episodes):
        t0 = time.perf_counter()
        sc = Scenario(cfg, seed * 1_000_000 + ep)
        rts = {arm: Runtime(cfg, sc) for arm in agents}
        for t in range(cfg.steps_per_episode):
            rng_before, rng_after = torch.get_rng_state(), None
            for i, (arm, agent) in enumerate(agents.items()):
                if i:
                    torch.set_rng_state(rng_before)
                rts[arm].step_agent(agent, t, LEARNING_ARMS[arm], training=True)
                agent.train_update()
                if i == 0:
                    rng_after = torch.get_rng_state()
            torch.set_rng_state(rng_after)
        for arm, agent in agents.items():
            m = rts[arm].metrics(arm, seed, sc_seed := seed * 1_000_000 + ep)
            rows.append({"arm": arm, "training_seed": seed, "episode": ep + 1,
                         "scenario_seed": sc_seed, "mean_episode_reward": m["mean_reward"],
                         "epsilon": agent.epsilon, "n_ho": m["n_ho"], "n_hof": m["n_hof"],
                         "n_rlf": m["n_rlf"], "outage_frac": m["outage_frac"]})
        last = rows[-len(agents):]
        print(f"seed {seed} ep {ep + 1}/{episodes} {time.perf_counter() - t0:.1f}s | " +
              " | ".join(f"{r['arm']}: R={r['mean_episode_reward']:.3f} ho={r['n_ho']} "
                         f"rlf={r['n_rlf']}" for r in last), flush=True)
    pd.DataFrame(rows).to_csv(out / "reward_curves.csv", index=False)
    for arm, agent in agents.items():
        agent.save(out / f"agent-{arm.replace(' ', '_').replace('=', '')}.pth")
    return agents, time.perf_counter() - start


def evaluate(cfg, seed, agents, baselines, out):
    rows = []
    start = time.perf_counter()
    a3s, a3t = baselines["static_a3"], baselines["tuned_a3"]
    for i in range(cfg.evaluation_scenarios):
        sc_seed = cfg.evaluation_seed_start + i
        sc = Scenario(cfg, sc_seed)
        for arm, agent in agents.items():
            rt = Runtime(cfg, sc)
            for t in range(cfg.steps_per_episode):
                rt.step_agent(agent, t, LEARNING_ARMS[arm], training=False)
            rows.append(rt.metrics(arm, seed, sc_seed))
        rt = Runtime(cfg, sc)
        for t in range(cfg.steps_per_episode):
            rt.step_rules(t)
        rows.append(rt.metrics(ARM_RULES, "n/a", sc_seed))
        for arm, p in ((ARM_A3, a3s), (ARM_A3T, a3t)):
            rt, ctl = Runtime(cfg, sc), A3Controller(cfg, p["margin_db"], p["ttt_s"])
            for t in range(cfg.steps_per_episode):
                ctl.step(rt, t)
            rows.append(rt.metrics(arm, "n/a", sc_seed))
        print(f"seed {seed} eval {i + 1}/{cfg.evaluation_scenarios} | " +
              " | ".join(f"{r['arm']}: ho={r['n_ho']} hof={r['n_hof']} rlf={r['n_rlf']} "
                         f"out={r['outage_frac']:.3f}" for r in rows[-6:]), flush=True)
    pd.DataFrame(rows).to_csv(out / "per_scenario.csv", index=False)
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-scenarios", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--phase", choices=["0", "0b", "0c"], default="0")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    cfg, baselines = load_config("config" if args.phase == "0" else f"config-{args.phase}")
    if args.steps:
        cfg = replace(cfg, steps_per_episode=args.steps)
    if args.eval_scenarios:
        cfg = replace(cfg, evaluation_scenarios=args.eval_scenarios)
    episodes = args.episodes or cfg.training_episodes
    raw = "raw" if args.phase == "0" else f"raw-{args.phase}"
    out = OUTPUT_ROOT / "logs" / raw / f"seed-{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    agents, train_s = train(cfg, args.seed, device, out, episodes)
    eval_s = evaluate(cfg, args.seed, agents, baselines, out)
    pd.DataFrame([{"training_seed": args.seed, "episodes": episodes,
                   "steps_per_episode": cfg.steps_per_episode,
                   "training_runtime_s": round(train_s, 1), "evaluation_runtime_s": round(eval_s, 1),
                   "threads": args.threads, "device": "CPU"}]).to_csv(out / "run-summary.csv",
                                                                       index=False)


if __name__ == "__main__":
    main()
