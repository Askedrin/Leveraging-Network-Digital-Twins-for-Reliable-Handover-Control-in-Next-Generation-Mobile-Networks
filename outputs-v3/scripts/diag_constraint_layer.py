#!/usr/bin/env python3
"""Phase 0b diagnostic (not pre-registered, not part of the go/no-go criterion).

Rules-only policy on the 20 evaluation scenarios with the constraint layer's parts switched
off one at a time (minimum interval, blacklist, 3-dB gain). No training.
Writes ../data/phase0b/diag-constraint-layer.csv.
"""

from dataclasses import replace

import pandas as pd

from run_seed import load_config
from twin import OUTPUT_ROOT, Runtime, Scenario

NO_BL = dict(blacklist_count_300s=10**6, blacklist_count_60s=10**6)
VARIANTS = {
    "full layer (30 s, blacklist, 3 dB)": {},
    "interval 0 s": dict(min_handover_interval_s=0.0),
    "no blacklist": NO_BL,
    "no blacklist, interval 1 s": dict(NO_BL, min_handover_interval_s=1.0),
    "no blacklist, interval 0 s": dict(NO_BL, min_handover_interval_s=0.0),
    "no blacklist, interval 0 s, gain 0 dB": dict(NO_BL, min_handover_interval_s=0.0,
                                                  min_rsrp_improvement_db=0.0),
}

cfg0, _ = load_config("config-0b")
rows = []
for i in range(cfg0.evaluation_scenarios):
    sc_seed = cfg0.evaluation_seed_start + i
    sc = Scenario(cfg0, sc_seed)
    for name, kw in VARIANTS.items():
        cfg = replace(cfg0, **kw)
        rt = Runtime(cfg, sc)
        for t in range(cfg.steps_per_episode):
            rt.step_rules(t)
        rows.append({"variant": name, **rt.metrics("Rules only", "n/a", sc_seed)})
    print(f"scenario {i + 1}/{cfg0.evaluation_scenarios}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_ROOT / "data" / "phase0b" / "diag-constraint-layer.csv", index=False)
print(df.groupby("variant", sort=False)[["ho_per_ueh", "hof_rate", "ppr1", "rlf_per_ueh",
                                    "outage_frac", "p5_sinr_db"]].mean().round(4).to_string())
