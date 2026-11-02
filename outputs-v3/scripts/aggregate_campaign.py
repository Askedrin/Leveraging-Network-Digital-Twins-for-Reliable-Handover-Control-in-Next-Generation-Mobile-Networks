#!/usr/bin/env python3
"""Aggregation and pre-registered tests of the Phase 1b + 2 campaign (../PREREGISTRATION-2.md).

Reads ../logs/raw-2/{seed-*,baselines,reference-*}/eval.csv. Learning arms are averaged over the
training seeds per scenario. Writes ../data/headline/, ../data/robustness/, ../data/mobility/,
../data/speed/, ../data/decomposition/ and ../data/training/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aggregate import DECISION_KPIS, bootstrap_ci, holm, paired_p
import campaign
from campaign import ARMS
from twin import OUTPUT_ROOT

KPIS = ["ho_per_ueh", "hsr", "hof_rate", "ppr1", "ppr10", "rlf_per_ueh", "outage_frac",
        "mean_sinr_db", "p5_sinr_db", "mean_rsrp_dbm", "rlf_too_late", "rlf_too_early",
        "rlf_wrong_cell", "rlf_coverage", "n_ho", "n_rlf", "n_hof", "n_pp1", "fallback_frac",
        "danger_frac"]
ORDER = ["Static A3", "Tuned A3", "MRO-A3", "Rules only, paper's layer", "Rules only",
         "Reactive DQN, paper's layer (Phase 0b)", "Oracle DQN k=5, paper's layer (Phase 0b)",
         "Reactive DQN, original reward (Phase 0c)", "Oracle DQN k=5, original reward (Phase 0c)",
         "Oracle DQN k=5, pre-registered design", "Proactive DQN, pre-registered design",
         "Proactive DQN + fallback, pre-registered design",
         "Reactive DQN", "Proactive DQN", "Proactive DQN + fallback", "Oracle DQN k=5"]
HYPOTHESES = [
    ("H1", "Proactive DQN + fallback", "Reactive DQN"),
    ("H2", "Proactive DQN", "Reactive DQN"),
    ("H3", "Proactive DQN + fallback", "Static A3"),
    ("H3", "Proactive DQN + fallback", "Tuned A3"),
    ("H3", "Proactive DQN + fallback", "MRO-A3"),
    ("H4", "Reactive DQN", "Rules only"),
    ("H5", "Rules only", "Rules only, paper's layer"),
]
H6 = [("H6", f"{a}", f"{a}, pre-registered design")
      for a in ("Proactive DQN + fallback", "Proactive DQN", "Oracle DQN k=5")]
REPORTED = [
    ("ref", "Oracle DQN k=5", "Reactive DQN"),
    ("ref", "Proactive DQN + fallback", "Proactive DQN"),
    ("ref", "Reactive DQN", "Static A3"),
    ("ref", "Reactive DQN", "Reactive DQN, paper's layer (Phase 0b)"),
    ("ref", "Reactive DQN", "Reactive DQN, original reward (Phase 0c)"),
    ("ref", "Oracle DQN k=5, original reward (Phase 0c)", "Reactive DQN, original reward (Phase 0c)"),
    ("ref", "Rules only", "Static A3"),
    ("ref", "Proactive DQN + fallback", "Rules only"),
    ("ref", "Oracle DQN k=5", "Rules only"),
]


def load():
    frames = []
    for p in sorted(campaign.RAW.glob("*/eval.csv")):
        d = pd.read_csv(p)
        d["source"] = p.parent.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for k in ("fallback_frac", "danger_frac"):
        if k not in df:
            df[k] = np.nan
    learned = df.source.str.startswith(("seed-", "reference-", "sensitivity-", "prereg-"))
    keys = ["condition", "variant", "arm", "scenario_seed"]
    per = pd.concat([
        df[learned].groupby(keys, as_index=False)[KPIS].mean(),
        df[~learned][keys + KPIS],
    ], ignore_index=True)
    return df, per


def summarise(per, cond, var):
    g = per[(per.condition == cond) & (per.variant == var)]
    rows = []
    for arm in [a for a in ORDER if a in set(g.arm)]:
        x = g[g.arm == arm].sort_values("scenario_seed")
        row = {"condition": cond, "variant": var, "arm": arm, "n_scenarios": len(x)}
        for k in KPIS:
            v = x[k].to_numpy(float)
            if np.isnan(v).all():
                continue
            v = v[~np.isnan(v)]
            lo, hi = bootstrap_ci(v)
            row[k], row[k + "_ci_lo"], row[k + "_ci_hi"] = v.mean(), lo, hi
        rows.append(row)
    return rows


def compare(per, cond, var, a, b, label):
    g = per[(per.condition == cond) & (per.variant == var)]
    A = g[g.arm == a].set_index("scenario_seed").sort_index()
    B = g[g.arm == b].set_index("scenario_seed").sort_index()
    common = A.index.intersection(B.index)
    if not len(common):
        return []
    A, B = A.loc[common], B.loc[common]
    recs, ps = [], []
    for k, sign in DECISION_KPIS.items():
        x, y = A[k].to_numpy(float), B[k].to_numpy(float)
        p = paired_p(x, y)
        ps.append(p)
        base = abs(y.mean())
        recs.append({"label": label, "condition": cond, "variant": var, "arm": a, "reference": b,
                     "kpi": k, "better": "higher" if sign > 0 else "lower",
                     "arm_mean": x.mean(), "ref_mean": y.mean(), "diff": x.mean() - y.mean(),
                     "relative_improvement": sign * (x.mean() - y.mean()) / base if base else np.nan,
                     "p_raw": p, "n": len(common)})
    for r, pa in zip(recs, holm(ps)):
        r["p_holm"] = pa
        r["better_sig"] = bool(pa < 0.05 and r["relative_improvement"] > 0)
        r["worse_sig"] = bool(pa < 0.05 and r["relative_improvement"] < 0)
    return recs


def verdict(recs):
    return any(r["better_sig"] for r in recs) and not any(r["worse_sig"] for r in recs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["2", "3"], default="2")
    study = ap.parse_args().study
    campaign.configure_study(study)
    suffix = "" if study == "2" else f"-{study}"
    raw, per = load()
    dirs = {k: OUTPUT_ROOT / "data" / f"{k}{suffix}" for k in
            ("headline", "robustness", "mobility", "speed", "decomposition", "training")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # headline (P1) and hypotheses
    head = pd.DataFrame(summarise(per, "headline", "main"))
    head.to_csv(dirs["headline"] / "summary.csv", index=False)
    per[per.condition == "headline"].to_csv(dirs["headline"] / "per_scenario.csv", index=False)
    tests, verdicts = [], {}
    for h, a, b in HYPOTHESES + (H6 if study == "3" else []) + REPORTED:
        recs = compare(per, "headline", "main", a, b, h)
        tests += recs
        if h != "ref" and recs:
            verdicts[f"{h}: {a} vs {b}"] = verdict(recs)
    pd.DataFrame(tests).to_csv(dirs["headline"] / "paired_tests.csv", index=False)
    seeds = sorted(int(s) for s in raw[raw.source.str.startswith("seed-")].training_seed.unique())
    seed_std = (raw[raw.source.str.startswith("seed-") & (raw.condition == "headline")]
                .groupby(["arm", "training_seed"])[list(DECISION_KPIS)].mean()
                .groupby("arm").std(ddof=1))
    seed_std.to_csv(dirs["headline"] / "seed_std.csv")
    (dirs["headline"] / "hypotheses.json").write_text(json.dumps(
        {"training_seeds": seeds, "scenarios": int(head.n_scenarios.max()),
         "verdicts": verdicts}, indent=2) + "\n")

    # sensitivity (DEVIATIONS.md D1): Transformer forecasts vs the headline CNN-LSTM forecasts
    if (per.condition == "sensitivity").any():
        sens = per[per.condition == "sensitivity"].copy()
        sens["arm"] = sens.arm + " (Transformer forecasts)"
        sens["condition"], sens["variant"] = "headline", "main"
        both = pd.concat([per[(per.condition == "headline")], sens], ignore_index=True)
        srows, stests = [], []
        for a in ("Proactive DQN", "Proactive DQN + fallback"):
            g = both[both.arm.isin([a, a + " (Transformer forecasts)", "Reactive DQN"])]
            stests += compare(g, "headline", "main", a + " (Transformer forecasts)", a, "D1")
            stests += compare(g, "headline", "main", a + " (Transformer forecasts)",
                              "Reactive DQN", "D1")
            for arm in (a + " (Transformer forecasts)",):
                x = g[g.arm == arm]
                srows.append({"arm": arm, **{k: x[k].mean() for k in KPIS}})
        sdir = OUTPUT_ROOT / "data" / f"sensitivity{suffix}"
        sdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(srows).to_csv(sdir / "summary.csv", index=False)
        pd.DataFrame(stests).to_csv(sdir / "paired_tests.csv", index=False)

    # decomposition (P15)
    dec = head[head.arm.isin(["Rules only, paper's layer", "Rules only",
                              "Reactive DQN, paper's layer (Phase 0b)", "Reactive DQN",
                              "Proactive DQN + fallback", "Oracle DQN k=5", "Static A3"])]
    dec.to_csv(dirs["decomposition"] / "rules_vs_dqn.csv", index=False)

    # robustness (P6): reactive and sigma=0 rows come from the headline subset
    sweep = sorted(set(per[per.condition == "noise"].scenario_seed))
    base = per[(per.condition == "headline") & per.scenario_seed.isin(sweep)].copy()
    base["condition"], base["variant"] = "noise", "sigma=0"
    noise = pd.concat([per[per.condition == "noise"], base], ignore_index=True)
    rows, tests = [], []
    variants = ["sigma=0"] + sorted(set(per[per.condition == "noise"].variant))
    ref = base[base.arm == "Reactive DQN"]
    for v in variants:
        g = pd.concat([noise[(noise.variant == v) & (noise.arm != "Reactive DQN")], ref.assign(variant=v)])
        g = g[g.arm.isin(["Reactive DQN", "Proactive DQN", "Proactive DQN + fallback"])]
        rows += summarise(g, "noise", v)
        for a in ("Proactive DQN", "Proactive DQN + fallback"):
            tests += compare(g, "noise", v, a, "Reactive DQN", "P6")
    pd.DataFrame(rows).to_csv(dirs["robustness"] / "kpi_vs_sigma.csv", index=False)
    pd.DataFrame(tests).to_csv(dirs["robustness"] / "paired_tests.csv", index=False)

    # mobility (P7) and speed (P8): the main variant is the headline subset
    for cond, main_label in (("mobility", "manhattan"), ("speed", "60-100")):
        b = per[(per.condition == "headline") & per.scenario_seed.isin(sweep)].copy()
        b["condition"], b["variant"] = cond, main_label
        g = pd.concat([per[per.condition == cond], b], ignore_index=True)
        rows, tests = [], []
        for v in [main_label] + sorted(set(per[per.condition == cond].variant)):
            rows += summarise(g, cond, v)
            for h, a, r in (("H1", "Proactive DQN + fallback", "Reactive DQN"),
                            ("H3", "Proactive DQN + fallback", "Static A3"),
                            ("ref", "Reactive DQN", "Static A3")):
                tests += compare(g, cond, v, a, r, h)
        pd.DataFrame(rows).to_csv(dirs[cond] / "summary.csv", index=False)
        pd.DataFrame(tests).to_csv(dirs[cond] / "paired_tests.csv", index=False)

    # training curves (P10)
    curves = [pd.read_csv(p) for p in sorted(campaign.RAW.glob("seed-*/reward_curves.csv"))]
    if curves:
        pd.concat(curves).to_csv(dirs["training"] / "reward_curves.csv", index=False)

    show = ["arm", "ho_per_ueh", "hof_rate", "ppr1", "rlf_per_ueh", "outage_frac", "p5_sinr_db",
            "mean_sinr_db"]
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", "{:.4g}".format):
        print(head[show].to_string(index=False))
        t = pd.read_csv(dirs["headline"] / "paired_tests.csv")
        print(t[["label", "arm", "reference", "kpi", "arm_mean", "ref_mean",
                 "relative_improvement", "p_holm"]].to_string(index=False))
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
