#!/usr/bin/env python3
"""Phase-0 aggregation and the pre-registered go/no-go test.

Reads ../logs/raw/seed-*/per_scenario.csv. Learning arms are averaged over training seeds per
scenario; seed-independent arms must be identical across seed runs (asserted).
Writes ../data/phase0/{summary,per_scenario,paired_tests,go_no_go}.csv and prints the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from twin import (ARM_A3, ARM_A3T, ARM_ORACLE5, ARM_ORACLE10, ARM_REACTIVE, ARM_RULES,
                  LEARNING_ARMS, OUTPUT_ROOT)

KPIS = ["hsr", "hof_rate", "ppr1", "ppr10", "rlf_per_ueh", "ho_per_ueh", "outage_frac",
        "mean_sinr_db", "p5_sinr_db", "mean_rsrp_dbm", "mean_reward", "n_ho", "n_hof", "n_rlf",
        "rlf_too_late", "rlf_too_early", "rlf_wrong_cell", "rlf_coverage", "n_pp1", "n_pp10"]
DECISION_KPIS = {"rlf_per_ueh": -1, "hof_rate": -1, "outage_frac": -1, "p5_sinr_db": +1}
ARM_ORDER = [ARM_A3, ARM_A3T, ARM_RULES, ARM_REACTIVE, ARM_ORACLE5, ARM_ORACLE10]


def bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_p(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["0", "0b", "0c"], default="0")
    phase = ap.parse_args().phase
    raw_dir = "raw" if phase == "0" else f"raw-{phase}"
    raw = sorted((OUTPUT_ROOT / "logs" / raw_dir).glob("seed-*/per_scenario.csv"))
    df = pd.concat([pd.read_csv(p) for p in raw], ignore_index=True)
    seeds = [int(s) for s in sorted(df.loc[df.arm.isin(LEARNING_ARMS), "training_seed"].astype(int).unique())]
    fixed = df[~df.arm.isin(LEARNING_ARMS)]
    for arm, g in fixed.groupby("arm"):
        per = g.groupby("scenario_seed")[KPIS].nunique()
        assert (per <= 1).all().all(), f"{arm} differs across seed runs"
    fixed = fixed.drop_duplicates(["arm", "scenario_seed"])
    learn = df[df.arm.isin(LEARNING_ARMS)]
    learn_mean = learn.groupby(["arm", "scenario_seed"], as_index=False)[KPIS].mean()
    per_scn = pd.concat([fixed[["arm", "scenario_seed"] + KPIS], learn_mean], ignore_index=True)

    out = OUTPUT_ROOT / "data" / f"phase{phase}"
    out.mkdir(parents=True, exist_ok=True)
    per_scn.to_csv(out / "per_scenario.csv", index=False)

    rows = []
    for arm in ARM_ORDER:
        g = per_scn[per_scn.arm == arm].sort_values("scenario_seed")
        row = {"arm": arm, "n_scenarios": len(g)}
        for k in KPIS:
            v = g[k].to_numpy(float)
            lo, hi = bootstrap_ci(v)
            row[k] = v.mean()
            row[k + "_ci_lo"], row[k + "_ci_hi"] = lo, hi
        if arm in LEARNING_ARMS:
            seed_means = learn[learn.arm == arm].groupby("training_seed")[KPIS].mean()
            for k in DECISION_KPIS:
                row[k + "_seed_std"] = float(seed_means[k].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    def pivot(arm):
        return per_scn[per_scn.arm == arm].sort_values("scenario_seed").set_index("scenario_seed")

    tests, verdicts = [], {}
    ref = pivot(ARM_REACTIVE)
    for arm in (ARM_ORACLE5, ARM_ORACLE10):
        cur = pivot(arm)
        ps, recs = [], []
        for k, sign in DECISION_KPIS.items():
            a, b = cur[k].to_numpy(float), ref[k].to_numpy(float)
            p = paired_p(a, b)
            base = abs(b.mean()) if b.mean() != 0 else np.nan
            rel = sign * (a.mean() - b.mean()) / base if base == base and base > 0 else np.nan
            ps.append(p)
            recs.append({"comparison": f"{arm} vs {ARM_REACTIVE}", "kpi": k,
                         "better_direction": "higher" if sign > 0 else "lower",
                         "arm_mean": a.mean(), "reactive_mean": b.mean(),
                         "relative_improvement": rel, "p_raw": p})
        for r, pa in zip(recs, holm(ps)):
            r["p_holm"] = pa
            r["improves"] = bool(pa < 0.05 and r["relative_improvement"] >= 0.10)
            r["degrades"] = bool(pa < 0.05 and r["relative_improvement"] < 0)
        tests += recs
        verdicts[arm] = any(r["improves"] for r in recs) and not any(r["degrades"] for r in recs)

    for a, b in ((ARM_RULES, ARM_REACTIVE), (ARM_A3T, ARM_A3)):
        pa, pb = pivot(a), pivot(b)
        for k, sign in DECISION_KPIS.items():
            x, y = pa[k].to_numpy(float), pb[k].to_numpy(float)
            tests.append({"comparison": f"{a} vs {b} (reported, not in criterion)", "kpi": k,
                          "better_direction": "higher" if sign > 0 else "lower",
                          "arm_mean": x.mean(), "reactive_mean": y.mean(),
                          "relative_improvement": np.nan, "p_raw": paired_p(x, y),
                          "p_holm": np.nan, "improves": np.nan, "degrades": np.nan})
    pd.DataFrame(tests).to_csv(out / "paired_tests.csv", index=False)
    go = any(verdicts.values())
    verdict = {"training_seeds": seeds, "scenarios": int(summary.n_scenarios.max()),
               "per_oracle_arm": verdicts, "verdict": "GO" if go else "NO-GO"}
    (out / "go_no_go.json").write_text(json.dumps(verdict, indent=2) + "\n")

    show = ["arm", "ho_per_ueh", "hsr", "hof_rate", "ppr1", "ppr10", "rlf_per_ueh",
            "outage_frac", "mean_sinr_db", "p5_sinr_db", "mean_rsrp_dbm"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4g}".format):
        print(summary[show].to_string(index=False))
        print()
        print(pd.DataFrame(tests)[["comparison", "kpi", "arm_mean", "reactive_mean",
                                   "relative_improvement", "p_raw", "p_holm"]].to_string(index=False))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
