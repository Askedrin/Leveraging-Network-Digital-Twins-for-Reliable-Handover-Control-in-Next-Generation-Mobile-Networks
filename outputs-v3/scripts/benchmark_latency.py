#!/usr/bin/env python3
"""P11: batch-1 decision latency of the coupled pipeline (PREREGISTRATION-2.md section 5).

One UE, one decision: forecast its 12 links (H = 15 window each), run the Q-network on its
state, apply the constraint layer (the state vector itself is built once, outside the timed
loop). 1 thread, 200 warm-up and 5,000 timed repetitions. Writes
../data/latency/{latency.csv,hardware.json}; with --arch, the forecaster row for that
architecture to latency-<arch>.csv (DEVIATIONS.md D1).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import campaign
from campaign import CampaignRuntime, base_config, configure_study, forecast_tools
from forecaster_twin import H, TwinForecaster, features
from twin import OUTPUT_ROOT, DQNAgent, Scenario, runtime_hardware

REPS, WARM = 5000, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["2", "3"], default="3")
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()
    study = args.study
    torch.set_num_threads(1)
    configure_study(study)
    cfg, _ = base_config()
    fcm, tau, sel = forecast_tools()
    agent = DQNAgent(cfg, 42, torch.device("cpu"))
    agent.load(campaign.RAW / "seed-42" / "agent-fallback.pth")
    sc = Scenario(cfg, campaign.HEADLINE[0])
    f = features(sc)
    rt = CampaignRuntime(cfg, sc, "fallback", fcm.predict(sc), tau)
    for t in range(200):
        rt.step_agent(agent, t, 5, training=False)
    t, v = 200, 0
    ring = [f[t - H + 1 + i, v] for i in range(H)]           # per-step feature ring buffer
    x = rt.source(t, 5)
    s, target = rt.states(t, x, 5)
    s1 = torch.from_numpy(s[v:v + 1])
    model, q = (TwinForecaster(args.arch) if args.arch else fcm).model, agent.q
    q.eval()
    srv, tgt = int(rt.serving[v]), int(target[v])
    times = {"forecaster": [], "agent": [], "constraint_layer": [], "end_to_end": []}
    with torch.no_grad():
        for i in range(WARM + REPS):
            t0 = time.perf_counter()
            win = torch.from_numpy(np.ascontiguousarray(np.stack(ring, axis=1)))   # (12, H, 12)
            fc = model(win)
            t1 = time.perf_counter()
            a = int(q(s1).argmax(dim=1).item())
            t2 = time.perf_counter()
            t_s = (t + 1) * cfg.dt_s
            sinr_srv = sc.sinr_meas[t, v, srv]
            override = sinr_srv < cfg.q_out_db + cfg.override_margin_db
            limits = ((t_s - rt.last_ho[v] >= cfg.min_handover_interval_s) and
                      (t_s >= rt.blacklist_until[v])) or override
            gain_ok = (x[v, tgt] - x[v, srv]) >= cfg.min_rsrp_improvement_db
            admit = sc.sinr_meas[t, v, tgt] >= cfg.q_in_db
            _ = (a == 1) and limits and gain_ok and admit
            t3 = time.perf_counter()
            if i >= WARM:
                times["forecaster"].append(t1 - t0)
                times["agent"].append(t2 - t1)
                times["constraint_layer"].append(t3 - t2)
                times["end_to_end"].append(t3 - t0)
    rows = []
    for k, arr in times.items():
        a = np.array(arr) * 1e3
        rows.append({"component": k, "p50_ms": np.percentile(a, 50), "p95_ms": np.percentile(a, 95),
                     "p99_ms": np.percentile(a, 99), "mean_ms": a.mean(), "reps": len(a)})
    out = OUTPUT_ROOT / "data" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if args.arch:
        df = df[df.component == "forecaster"].assign(arch=args.arch)
        df.to_csv(out / f"latency-{args.arch}.csv", index=False)
        print(df.round(4).to_string(index=False))
        return
    df.to_csv(out / "latency.csv", index=False)
    hw = runtime_hardware()
    hw.update({"device_used": "CPU", "study": study, "scenario": campaign.HEADLINE[0], "threads": 1,
               "batch": "1 UE (12 links)", "forecaster": sel["selected_arch"],
               "torch": torch.__version__})
    (out / "hardware.json").write_text(json.dumps(hw, indent=2, default=str) + "\n")
    print(df.round(4).to_string(index=False))
    print(json.dumps(hw, default=str))


if __name__ == "__main__":
    main()
