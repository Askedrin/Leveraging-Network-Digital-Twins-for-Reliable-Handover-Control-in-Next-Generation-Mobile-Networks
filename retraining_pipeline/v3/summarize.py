"""
v3/summarize.py
===============
Phase-1a table (P4): every trained model plus persistence and linear extrapolation, k = 1..5 s,
on the test traces, and the pre-registered selection (lowest validation MSE averaged over k,
across both target types). Writes results/summary.csv and results/selection.json.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5


def naive_rows(d, split):
    rf, rl, rp = d["r_future"], d["r_last"], d["r_prev"]
    rows = []
    for name in ("Persistence", "Linear extrapolation"):
        for k in range(K):
            p = rl if name == "Persistence" else rl + (k + 1) * (rl - rp)
            e, pe = rf[:, k] - p, rf[:, k] - rl
            ch = rf[:, k] != rl
            rows.append({"model": name, "target": "-", "split": split, "k": k + 1,
                         "MAE_dB": np.abs(e).mean(), "RMSE_dB": np.sqrt((e ** 2).mean()),
                         "R2": 1 - (e ** 2).sum() / ((rf[:, k] - rf[:, k].mean()) ** 2).sum(),
                         "skill_vs_persistence": 1 - (e ** 2).mean() / (pe ** 2).mean(),
                         "changed_frac": ch.mean(), "changed_MAE_dB": np.abs(e[ch]).mean()})
    return rows


def main():
    rows, val_mse = [], {}
    for f in sorted(glob.glob(os.path.join(HERE, "results", "*_*.json"))):
        r = json.load(open(f))
        if "arch" not in r:
            continue
        val_mse[f"{r['arch']}_{r['target']}"] = r["val_mse_db_mean_over_k"]
        for split in ("val", "test"):
            for m in r[split]:
                rows.append({"model": r["arch"], "target": r["target"], "split": split,
                             "k": m["k"], "MAE_dB": m["MAE_dB"], "RMSE_dB": m["RMSE_dB"],
                             "R2": m["R2"], "skill_vs_persistence": m["skill_vs_persistence"],
                             "changed_frac": m["changed_frac"],
                             "changed_MAE_dB": m["changed_MAE_dB"], "params": r["params"],
                             "epochs_run": r["epochs_run"]})
    for split in ("val", "test"):
        rows += naive_rows(np.load(os.path.join(HERE, "data", f"{split}.npz")), split)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "summary.csv"), index=False)
    chosen = min(val_mse, key=val_mse.get)
    sel = {"selected": chosen, "rule": "lowest validation MSE averaged over k (both targets)",
           "val_mse_db2": dict(sorted(val_mse.items(), key=lambda kv: kv[1])),
           "n_models": len(val_mse), "complete": len(val_mse) == 12}
    json.dump(sel, open(os.path.join(HERE, "results", "selection.json"), "w"), indent=2)
    t = df[df.split == "test"].pivot_table(index=["model", "target"], columns="k",
                                           values=["MAE_dB", "skill_vs_persistence"])
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(t)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
