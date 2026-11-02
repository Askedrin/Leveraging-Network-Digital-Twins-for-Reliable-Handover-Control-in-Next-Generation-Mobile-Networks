#!/usr/bin/env python3
"""Vector-PDF figures for the manuscript, each with the CSV of the plotted data.

  python figures.py [twin] [layout] [forecast-real] [training] [ablation] [robustness]
                    [rsrp-time] [kpi]

forecast-real, training, rsrp-time and kpi redraw the four result figures of the submitted
paper in their original form with the new data.

Writes ../render/*.pdf and ../render/*.csv. Palette: reference categorical order (blue,
orange, aqua, yellow, ...), assigned per entity and never cycled; markers and line styles
carry identity in grayscale print.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from twin import OUTPUT_ROOT  # noqa: E402

RENDER = OUTPUT_ROOT / "render"
DATA = OUTPUT_ROOT / "data"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ENTITY = {   # fixed entity -> (colour slot, marker, line style)
    "Reactive DQN": (0, "o", "-"), "Proactive DQN": (1, "s", "--"),
    "Proactive DQN + fallback": (2, "^", "-."), "Oracle DQN k=5": (3, "D", ":"),
    "Static A3": (4, "v", "-"), "Tuned A3": (5, "<", "--"), "MRO-A3": (6, ">", "-."),
    "Rules only": (7, "x", ":"),
    "Measured": (0, None, "-"), "Selected model": (1, None, "--"), "Persistence": (2, None, ":"),
}
SHORT = {"Reactive DQN": "Measurement-only", "Proactive DQN": "Proactive",
         "Proactive DQN + fallback": "Proactive + fallback", "Oracle DQN k=5": "Oracle (1 s)"}
COL_W, TEXT_W = 3.45, 7.16
SUFFIX = os.environ.get("STUDY_SUFFIX", "")          # "" = Study 2, "-3" = Study 3


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "legend.fontsize": 6.8, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.grid": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def minus(text):
    return text.replace("-", "\u2212")


def color(name):
    return SLOT[ENTITY[name][0]]


def save(fig, name, df):
    RENDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(RENDER / f"{name}.pdf")
    df.to_csv(RENDER / f"{name}.csv", index=False)
    plt.close(fig)
    print("wrote", RENDER / f"{name}.pdf")


def fig_twin():
    cdf = pd.read_csv(DATA / "twin-validation" / "cdf.csv")
    labels = {"coupling_loss_db": "Coupling loss (dB)",
              "best_rsrp_dbm_per_re": "RSRP (dBm/RE)",
              "best_cell_sinr_db": "SINR (dB)"}
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 1.35), sharey=True)
    fig.subplots_adjust(wspace=0.3)
    for ax, (k, lab) in zip(axes, labels.items()):
        g = cdf[cdf.metric == k]
        ax.plot(g.value, g.cdf, color=SLOT[0])
        ax.set_xlabel(lab)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    axes[0].set_ylabel("CDF")
    axes[0].set_ylim(0, 1)
    save(fig, "twin-cdfs", cdf)


def fig_layout():
    cells = pd.read_csv(DATA / "layout" / "cells.csv")
    traj = pd.read_csv(DATA / "layout" / "example-trajectories.csv")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
    for x in np.arange(0, 2001, 250):
        ax.axvline(x, color=GRID, lw=0.4, zorder=0)
    for y in np.arange(0, 1501, 250):
        ax.axhline(y, color=GRID, lw=0.4, zorder=0)
    ax.grid(False)
    for ue, g in traj.groupby("ue"):
        ax.plot(g.x_m, g.y_m, color=INK2, lw=0.8, alpha=0.7,
                label="Example trajectories" if ue == 0 else None)
    marks = {"macro": ("^", 46, SLOT[0]), "micro": ("s", 26, SLOT[1]), "pico": ("o", 22, SLOT[2])}
    for kind, (m, s, c) in marks.items():
        g = cells[cells.type == kind]
        ax.scatter(g.x_m, g.y_m, marker=m, s=s, color=c, edgecolor="white", linewidth=0.8,
                   zorder=3, label=f"{kind.capitalize()} cell")
    for _, r in cells.iterrows():
        off = (-4, -10) if r.type == "pico" else (3, 3)
        ax.annotate(str(int(r.cell)), (r.x_m, r.y_m), xytext=off, textcoords="offset points",
                    fontsize=6, color=INK2)
    ax.set_xlim(0, 2000), ax.set_ylim(0, 1500), ax.set_aspect("equal")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, handletextpad=0.3,
              columnspacing=0.8)
    save(fig, "layout", cells)


def fig_forecast_real():
    """Original forecast figure: two forecasters vs measured RSRP on a held-out ~6-min drive."""
    import torch
    sys.path.insert(0, str(OUTPUT_ROOT.parent / "retraining_pipeline"))
    from architectures import build_model
    v3 = OUTPUT_ROOT.parent / "retraining_pipeline" / "v3"
    acc = pd.read_csv(v3 / "results" / "summary.csv")
    d = np.load(v3 / "data" / "test.npz")
    k = 3                                                          # 3-s horizon (index 2)
    seg_ids, counts = np.unique(d["seg"], return_counts=True)
    idx = np.flatnonzero(d["seg"] == seg_ids[np.argmin(np.abs(counts - 360))])
    acc = acc[(acc.split == "test") & (acc.k == k)]
    pers = acc[acc.model == "Persistence"].iloc[0]
    rows = {"t_s": np.arange(len(idx)) + k, "measured_dbm": d["r_future"][idx, k - 1],
            "persistence_dbm": d["r_last"][idx]}
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_W, 2.9), sharex=True)
    fig.subplots_adjust(hspace=0.32)
    for ax, (arch, name) in zip(axes, (("LSTMAttention", "LSTM-Attention"), ("TCN", "TCN"))):
        model = build_model(arch, 12, 5)        # change target: lower validation error for both
        model.load_state_dict(torch.load(v3 / "checkpoints" / f"{arch}_delta.pth",
                                         weights_only=True))
        model.eval()
        with torch.no_grad():
            p = model(torch.from_numpy(d["X"][idx])).numpy()
        rows[f"{arch}_dbm"] = d["r_last"][idx] + p[:, k - 1] * 96
        m = acc[(acc.model == arch) & (acc.target == "delta")].iloc[0]
        ax.plot(rows["t_s"], rows["measured_dbm"], color=color("Measured"), lw=1.0,
                label="Measured RSRP")
        ax.plot(rows["t_s"], rows[f"{arch}_dbm"], color=color("Selected model"), ls="--", lw=1.0,
                label=f"Model forecast, {k} s ahead")
        ax.plot(rows["t_s"], rows["persistence_dbm"], color=color("Persistence"), ls=":", lw=1.0,
                label=f"Persistence, {k} s ahead")
        ax.set_title(f"{name}: MAE {m.MAE_dB:.2f} dB, RMSE {m.RMSE_dB:.2f} dB, $R^2$ = {m.R2:.3f}"
                     f"  (persistence: MAE {pers.MAE_dB:.2f} dB, RMSE {pers.RMSE_dB:.2f} dB, "
                     f"$R^2$ = {pers.R2:.3f})", loc="left")
        ax.set_ylabel("RSRP (dBm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(0, rows["t_s"][-1] + 1)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, handlelength=2.2,
                   columnspacing=1.5)
    save(fig, "forecast-real", pd.DataFrame(rows))


def fig_training():
    """Original convergence figure: reward of the proposed agent per training episode."""
    rc = pd.read_csv(DATA / f"training{SUFFIX}" / "reward_curves.csv")
    arm = "Proactive DQN + fallback"
    g = rc[rc.arm == arm].pivot(index="episode", columns="training_seed",
                                values="mean_episode_reward")
    mean = g.mean(axis=1)
    ma = mean.rolling(10, min_periods=10).mean()
    conv = mean.loc[81:].mean()                                    # last 20 episodes
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    ax.fill_between(g.index, g.min(axis=1), g.max(axis=1), color=color(arm), alpha=0.15, lw=0,
                    label="Range over 5 seeds")
    ax.plot(mean.index, mean, color=color(arm), lw=0.6, alpha=0.7, label="Mean over seeds")
    ax.plot(ma.index, ma, color=color(arm), lw=1.6, label="Moving average (10 episodes)")
    ax.axhline(conv, color=INK2, ls="--", lw=0.8, label=f"Converged ({minus(f'{conv:.3f}')})")
    ax.set_xlabel("Training episode"), ax.set_ylabel("Mean reward per UE-step")
    ax.set_xlim(0, 101)
    ax.legend(loc="lower right", ncol=1, handlelength=2.0)
    save(fig, f"training{SUFFIX}", pd.DataFrame({"episode": g.index, "mean_over_seeds": mean,
                                                 "min_seed": g.min(axis=1),
                                                 "max_seed": g.max(axis=1),
                                                 "moving_average_10": ma, "converged": conv}))


def fig_ablation():
    """Ablation of 'forecast, then decide safely': RLF of A3, no forecast, proposed and
    forecast-first (one bar each)."""
    s = pd.read_csv(DATA / f"headline{SUFFIX}" / "summary.csv").set_index("arm")
    bars = [("Static A3", "Static A3", "A3", False),
            ("Reactive DQN", "Reactive DQN", "No forecast", False),
            ("Proactive DQN + fallback", "Proactive DQN + fallback", "Proposed", False),
            ("Proactive DQN + fallback, pre-registered design", "Proactive DQN + fallback",
             "Forecast-first", True)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.35))
    rows = []
    for i, (arm, ent, lab, ff) in enumerate(bars):
        r = s.loc[arm]
        m, lo, hi = r.rlf_per_ueh, r.rlf_per_ueh_ci_lo, r.rlf_per_ueh_ci_hi
        ax.barh(i, m, height=0.62, color=color(ent), hatch="////" if ff else None,
                edgecolor="white", linewidth=0.0, zorder=2)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color=INK2, lw=0.8, capsize=1.5, zorder=3)
        ax.annotate(f"{m:.2f}" if m < 10 else f"{m:.1f}", (hi, i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5)
        rows.append({"method": lab, "rlf_per_ueh": m, "ci_lo": lo, "ci_hi": hi})
    ax.axhline(0.5, color=INK2, lw=0.6, ls="--")
    ax.set_yticks(range(len(bars)), [b[2] for b in bars])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 19.5)
    ax.set_xlabel("RLF per UE-hour")
    save(fig, f"ablation{SUFFIX}", pd.DataFrame(rows))


def fig_robustness():
    """RLF vs forecast noise: (a) pre-registered design (Study 2), (b) reliability-first (Study 3)."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.65), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    rows = []
    labels = {"Reactive DQN": "No forecast", "Proactive DQN": "Without fallback",
              "Proactive DQN + fallback": "With fallback"}
    for ax, (suffix, title) in zip(axes, (("", "(a) Forecast-first"),
                                         ("-3", "(b) Proposed"))):
        s = pd.read_csv(DATA / f"robustness{suffix}" / "kpi_vs_sigma.csv")
        sig = s[s.variant.str.startswith("sigma")].copy()
        sig["sigma"] = sig.variant.str.split("=").str[1].astype(float)
        for arm in ["Reactive DQN", "Proactive DQN", "Proactive DQN + fallback"]:
            g = sig[sig.arm == arm].sort_values("sigma")
            ax.plot(g.sigma, g.rlf_per_ueh, color=color(arm), ls=ENTITY[arm][2],
                    marker=ENTITY[arm][1], ms=3.5, label=labels[arm])
            ax.fill_between(g.sigma, g.rlf_per_ueh_ci_lo, g.rlf_per_ueh_ci_hi,
                            color=color(arm), alpha=0.12, lw=0)
            rows += [{"design": title, "arm": arm, "sigma": r.sigma, "rlf_per_ueh": r.rlf_per_ueh,
                      "outage_frac": r.outage_frac, "fallback_frac": r.fallback_frac}
                     for r in g.itertuples()]
        ax.set_title(title)
        ax.set_xlabel("Forecast noise $\\sigma$ (dB)")
    axes[0].set_ylabel("RLF per UE-hour")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.3), ncol=3, handlelength=2.2,
                   columnspacing=0.8)
    save(fig, "robustness", pd.DataFrame(rows))


def fig_rsrp_time():
    """Original RSRP-evolution figure: mean serving RSRP over the 200-s evaluation episode."""
    r = pd.read_csv(DATA / "rsrp-trace-3" / "rsrp_vs_time.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    bounds = [-70, -85, -100, -110]                              # classes of s6 (Table IV)
    for y in bounds:
        ax.axhline(y, color=INK2, ls="--", lw=0.6)
    for y, name in ((-67.5, "Excellent"), (-77.5, "Good"), (-92.5, "Fair"), (-105, "Poor"),
                    (-112.5, "Very poor")):
        ax.text(203, y, name, ha="left", va="center", fontsize=6.5, color=INK2, clip_on=False)
    for arm, lab in (("Static A3", "A3"), ("Proactive DQN + fallback", "Proposed")):
        g = r[r.arm == arm]
        avg = minus(f"{g['mean'].mean():.1f}")
        ax.plot(g.time_s, g["mean"], color=color(arm), ls=ENTITY[arm][2], lw=1.1,
                label=f"{lab} (mean {avg} dBm)")
    ax.set_xlim(0, 200), ax.set_ylim(-115, -65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time (s)"), ax.set_ylabel("Mean serving RSRP (dBm)")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.75), ncol=1, handlelength=2.2)
    save(fig, "rsrp-time", r)


def fig_kpi():
    """Original KPI and handover-volume comparison: static A3 vs the proposed design.

    The handover success rate of the original figure is left out: it is the same for both
    (99.83 %) and is reported in the main table."""
    s = pd.read_csv(DATA / "headline-3" / "summary.csv").set_index("arm")
    arms = (("Static A3", "Static A3"), ("Proactive DQN + fallback", "Proposed"))
    n = s.loc["Static A3", "n_scenarios"]
    panels = (("ppr1", 100, "(a) Ping-pong (%)", "{:.1f}"),
              ("rlf_per_ueh", 1, "(b) RLF per UE-h", "{:.2f}"),
              ("n_ho", n / 1000, "(c) Handovers (k)", "{:.1f}"))
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 1.6))
    fig.subplots_adjust(wspace=0.55)
    rows = []
    for ax, (k, f, title, fmt) in zip(axes, panels):
        for i, (arm, lab) in enumerate(arms):
            v, lo, hi = (s.loc[arm, c] * f for c in (k, k + "_ci_lo", k + "_ci_hi"))
            ax.bar(i, v, width=0.6, color=color(arm), zorder=2)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK2, lw=0.8, capsize=2, zorder=3)
            ax.annotate(fmt.format(v), (i, hi), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5)
            rows.append({"kpi": k, "arm": arm, "value": v, "ci_lo": lo, "ci_hi": hi})
        ax.set_xticks([0, 1], ["A3", "Prop."])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    save(fig, "kpi", pd.DataFrame(rows))


FIGS = {"twin": fig_twin, "layout": fig_layout, "forecast-real": fig_forecast_real,
        "training": fig_training, "ablation": fig_ablation, "robustness": fig_robustness,
        "rsrp-time": fig_rsrp_time, "kpi": fig_kpi}


if __name__ == "__main__":
    style()
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
