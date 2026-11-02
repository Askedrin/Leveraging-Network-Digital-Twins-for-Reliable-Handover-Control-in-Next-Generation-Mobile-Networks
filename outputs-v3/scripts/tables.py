#!/usr/bin/env python3
"""LaTeX tables for the manuscript, generated from the result CSVs (no hand-typed numbers).

Writes resubmission/revised/latex/tables/{res-main,res-forecast,res-mobility}.tex.
Method names follow the manuscript: A3, Tuned A3, A3 + MRO, No forecast, Proposed,
Forecast-first. The rule-only and perfect-forecast (oracle) arms are reported in REPORT.md,
not in the paper.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from twin import OUTPUT_ROOT

DATA = OUTPUT_ROOT / "data"
V3 = OUTPUT_ROOT.parent / "retraining_pipeline" / "v3"
OUT = OUTPUT_ROOT.parent / "resubmission" / "revised" / "latex" / "tables"


def ci(r, k, f=1.0, d=1):
    return (f"{r[k] * f:.{d}f} {{\\scriptsize[{r[k + '_ci_lo'] * f:.{d}f}, "
            f"{r[k + '_ci_hi'] * f:.{d}f}]}}")


def res_main():
    """Final design (Study 3, scenarios 3100-3149). CI only for the RLF rate."""
    s = pd.read_csv(DATA / "headline-3" / "summary.csv").set_index("arm")
    groups = [("3GPP baselines", [("Static A3", "A3"), ("Tuned A3", "Tuned A3"),
                                  ("MRO-A3", "A3 + MRO")]),
              ("Learned (DQN)", [("Reactive DQN", "No forecast"),
                                 ("Proactive DQN + fallback", r"\textbf{Proposed}"),
                                 ("Proactive DQN + fallback, pre-registered design",
                                  "Forecast-first")])]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{HO/UE-h} & \textbf{HSR (\%)} & \textbf{PPR (\%)} & "
             r"\textbf{RLF/UE-h [95\% CI]} & \textbf{Outage (\%)} & "
             r"\textbf{5th-pct SINR (dB)} \\"]
    for title, rows in groups:
        lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{" + title + r"}} \\"]
        for arm, name in rows:
            r = s.loc[arm]
            lines.append(f"{name} & {r.ho_per_ueh:.0f} & {100 * r.hsr:.2f} & "
                         f"{100 * r.ppr1:.1f} & {ci(r, 'rlf_per_ueh', d=2)} & "
                         f"{100 * r.outage_frac:.2f} & {r.p5_sinr_db:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def res_forecast():
    """RMSE of the basic model: real traces at k = 1, 3, 5 s and the simulated network at 1 s.

    Real traces: per architecture, the target with the lower validation MSE."""
    s = pd.read_csv(V3 / "results" / "summary.csv")
    sel = json.loads((V3 / "results" / "selection.json").read_text())
    t = s[s.split == "test"]
    val = {}
    for f in V3.glob("results/*_*.json"):
        r = json.loads(f.read_text())
        if "arch" in r:
            val[(r["arch"], r["target"])] = r["val_mse_db_mean_over_k"]
    a = pd.read_csv(DATA / "predictor-twin" / "accuracy.csv")
    twin_sel = json.loads((DATA / "predictor-twin" / "selection.json").read_text())
    tw = a[a.split == "test"].set_index("arch")
    label = {"ImprovedLSTM": "Improved LSTM", "LSTMAttention": "LSTM-Attention",
             "CNNLSTM": "CNN-LSTM"}

    def real(model, target):
        g = t[(t.model == model) & (t.target == target)].set_index("k")
        return " & ".join(f"{g.loc[k, 'RMSE_dB']:.2f}" for k in (1, 3, 5))

    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{\textbf{Real traces}} & \textbf{Simulated} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-5}",
             r"\textbf{Forecast} & $k{=}1$~s & 3~s & 5~s & $k{=}1$~s \\", r"\midrule",
             f"Persistence & {real('Persistence', '-')} & "
             f"{tw.iloc[0].persistence_RMSE_dB:.2f} \\\\",
             f"Linear extrapolation & {real('Linear extrapolation', '-')} & "
             f"{tw.iloc[0].linear_RMSE_dB:.2f} \\\\", r"\midrule"]
    for m in ["ImprovedLSTM", "BiLSTM", "LSTMAttention", "TCN", "Transformer", "CNNLSTM"]:
        tg = min((k for k in val if k[0] == m), key=val.get)[1]
        mark = ("$^*$" if f"{m}_{tg}" == sel["selected"] else "") + \
               ("$^\\dagger$" if m == twin_sel["selected_arch"] else "") + \
               ("$^\\ddagger$" if (m, "abs") not in val else "")
        lines.append(f"{label.get(m, m)}{mark} & {real(m, tg)} & {tw.loc[m, 'RMSE_dB']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def res_mobility():
    """Static A3 vs the proposed design per mobility model and speed; * = significant."""
    a, b = "Static A3", "Proactive DQN + fallback"
    kpis = [("ho_per_ueh", 1, 0, None), ("rlf_per_ueh", 1, 2, "rlf_per_ueh"),
            ("outage_frac", 100, 2, "outage_frac"), ("p5_sinr_db", 1, 1, "p5_sinr_db")]
    lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{\textbf{HO/UE-h}} & \multicolumn{2}{c}{\textbf{RLF/UE-h}} & "
             r"\multicolumn{2}{c}{\textbf{Outage (\%)}} & "
             r"\multicolumn{2}{c}{\textbf{5th-pct SINR (dB)}} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
             r"\textbf{Condition} & A3 & Prop. & A3 & Prop. & A3 & Prop. & A3 & Prop. \\",
             r"\midrule"]
    vlabel = {"manhattan": "Manhattan, 60--100 km/h (training)", "random_walk": "Random walk",
              "random_waypoint": "Random waypoint", "gauss_markov": "Gauss--Markov",
              "10-60": "Manhattan, 10--60 km/h", "30-60": "Manhattan, 30--60 km/h",
              "100-130": "Manhattan, 100--130 km/h"}
    for cond, variants in (("mobility", ["manhattan", "random_walk", "random_waypoint",
                                         "gauss_markov"]),
                           ("speed", ["10-60", "30-60", "100-130"])):
        s = pd.read_csv(DATA / f"{cond}-3" / "summary.csv")
        tests = pd.read_csv(DATA / f"{cond}-3" / "paired_tests.csv")
        tests = tests[(tests.arm == b) & (tests.reference == a)]
        for v in variants:
            g = s[s.variant == v].set_index("arm")
            cells = []
            for k, f, d, test in kpis:
                sig = ""
                if test:
                    row = tests[(tests.variant == v) & (tests.kpi == test)]
                    sig = "$^*$" if (row.better_sig | row.worse_sig).any() else ""
                cells += [f"{g.loc[a, k] * f:.{d}f}", f"{g.loc[b, k] * f:.{d}f}{sig}"]
            lines.append(f"{vlabel[v]} & " + " & ".join(cells) + r" \\")
        if cond == "mobility":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {"res-main": res_main, "res-forecast": res_forecast, "res-mobility": res_mobility}
    for name in (sys.argv[1:] or jobs):
        (OUT / f"{name}.tex").write_text(jobs[name]())
        print("wrote", OUT / f"{name}.tex")


if __name__ == "__main__":
    main()
