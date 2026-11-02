#!/usr/bin/env python3
"""Phase 1b: the in-twin basic model (RSRP forecaster), PREREGISTRATION-2.md section 1.

  python forecaster_twin.py data                 build ../data/predictor-twin/{train,val,test}.npz
  python forecaster_twin.py train --arch TCN     train one architecture (direct k = 5 delta)
  python forecaster_twin.py select               pick by validation MSE, compute tau_e, write
                                                 ../data/predictor-twin/{accuracy.csv,selection.json}

TwinForecaster(arch).predict(sc) gives, for every step t, the forecast of the L3 RSRP of every
UE-cell link at t + K, issued at t (persistence for the first H - 1 steps).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

from run_seed import load_config
from twin import OUTPUT_ROOT, Scenario, cell_kind_index

sys.path.insert(0, str(OUTPUT_ROOT.parent / "retraining_pipeline"))
import config as C                                          # noqa: E402
from architectures import build_model                       # noqa: E402

ARCHS = ["ImprovedLSTM", "TCN", "CNNLSTM", "LSTMAttention", "BiLSTM", "Transformer"]
H, K, N_FEAT = 15, 5, 12
SPAN = 96.0
SPLITS = {"train": range(8_000_000, 8_000_020), "val": range(8_000_020, 8_000_025),
          "test": range(8_000_025, 8_000_030)}
PER_SCENARIO = 3000
MAX_EPOCHS, PATIENCE, LR_PATIENCE = 30, 5, 3
W_ERR = 25                       # realised-error window (5 s)
DATA = OUTPUT_ROOT / "data" / "predictor-twin"
CKPT = OUTPUT_ROOT / "checkpoints" / "twin-forecaster"


def campaign_config():
    return load_config("config-0c")[0]


def features(sc: Scenario) -> np.ndarray:
    """(steps, nv, nc, 12) policy-independent link features (PREREGISTRATION-2.md)."""
    cfg = sc.cfg
    r = sc.meas
    steps, nv, nc = r.shape
    b = lambda a: np.broadcast_to(a[:, :, None], (steps, nv, nc))
    hd = np.radians(sc.heading)
    bearing = np.arctan2(sc.cells_y[None, None, :] - sc.y[:, :, None],
                         sc.cells_x[None, None, :] - sc.x[:, :, None])
    rel = hd[:, :, None] - bearing
    trend = np.zeros_like(r)
    trend[1:] = (r[1:] - r[:-1]) / 10.0
    ddist = np.zeros_like(r)
    step_m = np.maximum(sc.speed * cfg.dt_s / 3.6, 1e-3)
    ddist[1:] = (sc.dist[1:] - sc.dist[:-1]) / step_m[1:, :, None]
    kind = np.array([cell_kind_index(c) for c in sc.cells]) / 2.0
    f = np.stack([
        b(sc.x / cfg.grid_width_m), b(sc.y / cfg.grid_height_m), b(sc.speed / 100.0),
        (r + 140.0) / SPAN, np.minimum(sc.dist / 2000.0, 1.5),
        b(np.sin(hd)), b(np.cos(hd)), np.cos(rel), np.sin(rel),
        np.clip(trend, -3, 3), np.broadcast_to(kind[None, None, :], (steps, nv, nc)),
        np.clip(ddist, -1.5, 1.5),
    ], axis=-1)
    return f.astype(np.float32)


def windows(f: np.ndarray) -> np.ndarray:
    """(steps - H + 1, nv, nc, H, 12): the window issued at t = index + H - 1."""
    return sliding_window_view(f, H, axis=0).transpose(0, 1, 2, 4, 3)


def build_data():
    cfg = campaign_config()
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    for name, seeds in SPLITS.items():
        X, y, r_now, r_fut, r_prev = [], [], [], [], []
        for s in seeds:
            sc = Scenario(cfg, s)
            f, r = features(sc), sc.meas
            t_hi = cfg.steps_per_episode - 1            # issue times t in [H-1, t_hi]
            t = np.arange(H - 1, t_hi + 1)
            cand = np.argwhere(r[t] >= cfg.r_min_dbm)   # (i, v, c), t = t[i]
            pick = cand[rng.choice(len(cand), PER_SCENARIO, replace=False)]
            ti, v, c = t[pick[:, 0]], pick[:, 1], pick[:, 2]
            X.append(np.stack([f[tt - H + 1:tt + 1, vv, cc] for tt, vv, cc in zip(ti, v, c)]))
            y.append(((r[ti + K, v, c] - r[ti, v, c]) / SPAN)[:, None])
            r_now.append(r[ti, v, c]), r_fut.append(r[ti + K, v, c]), r_prev.append(r[ti - 1, v, c])
        np.savez_compressed(DATA / f"{name}.npz", X=np.concatenate(X).astype(np.float32),
                            y=np.concatenate(y).astype(np.float32), r_now=np.concatenate(r_now),
                            r_fut=np.concatenate(r_fut), r_prev=np.concatenate(r_prev))
        print(name, sum(len(a) for a in X), "windows", flush=True)


def metrics(pred_db, d):
    t, rn, rp = d["r_fut"], d["r_now"], d["r_prev"]
    out = {}
    for label, p in (("model", pred_db), ("persistence", rn), ("linear", rn + K * (rn - rp))):
        e = t - p
        ch = np.abs(t - rn) > 1.0
        out[label] = {"MAE_dB": float(np.abs(e).mean()), "RMSE_dB": float(np.sqrt((e ** 2).mean())),
                      "R2": float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum()),
                      "changed_frac": float(ch.mean()),
                      "changed_MAE_dB": float(np.abs(e[ch]).mean())}
    out["skill_vs_persistence"] = float(1 - ((t - pred_db) ** 2).mean() / ((t - rn) ** 2).mean())
    return out


def predict_array(model, X, batch=8192):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))).numpy()
                               for i in range(0, len(X), batch)])


def train(arch: str, threads: int):
    torch.set_num_threads(threads)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    CKPT.mkdir(parents=True, exist_ok=True)
    tr, va, te = (np.load(DATA / f"{n}.npz") for n in ("train", "val", "test"))
    model = build_model(arch, N_FEAT, 1)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LR_PATIENCE,
                                                       factor=C.LR_FACTOR)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"])),
                        batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(C.SEED))
    best, best_state, bad, t0, epochs = np.inf, None, 0, time.perf_counter(), 0
    for epoch in range(MAX_EPOCHS):
        epochs = epoch + 1
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
        vloss = float(((predict_array(model, va["X"]) - va["y"]) ** 2).mean())
        sched.step(vloss)
        if vloss < best:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{arch}] epoch {epochs} val_mse={vloss:.7f} {time.perf_counter() - t0:.0f}s",
              flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CKPT / f"{arch}.pth")
    to_db = lambda d: d["r_now"] + predict_array(model, d["X"])[:, 0] * SPAN
    out = {"arch": arch, "params": sum(p.numel() for p in model.parameters()),
           "epochs_run": epochs, "train_seconds": round(time.perf_counter() - t0, 1),
           "val_mse_norm": best, "val": metrics(to_db(va), va), "test": metrics(to_db(te), te)}
    (DATA / f"result-{arch}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"arch": arch, "test_MAE": round(out["test"]["model"]["MAE_dB"], 3),
                      "persistence_MAE": round(out["test"]["persistence"]["MAE_dB"], 3),
                      "skill": round(out["test"]["skill_vs_persistence"], 3)}), flush=True)


class TwinForecaster:
    def __init__(self, arch: str, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.arch = arch
        self.model = build_model(arch, N_FEAT, 1)
        self.model.load_state_dict(torch.load(CKPT / f"{arch}.pth", weights_only=True))
        self.model.eval()

    def predict(self, sc: Scenario) -> np.ndarray:
        """(steps, nv, nc) forecast of meas[t + K] issued at t."""
        f, r = features(sc), sc.meas
        w = windows(f)                                   # (steps-H+1, nv, nc, H, 12)
        n = w.shape[0]
        flat = w.reshape(-1, H, N_FEAT)
        d = predict_array(self.model, flat)[:, 0].reshape(n, *r.shape[1:])
        out = r.copy()
        out[H - 1:] = r[H - 1:] + d * SPAN
        return out.astype(np.float32)


def realised_error(fc: np.ndarray, meas: np.ndarray, r_min: float, steps: int) -> np.ndarray:
    """(steps, nv) rolling W_ERR mean of the per-step mean |forecast(t-K) - meas(t)| over
    candidate cells, as defined in PREREGISTRATION-2.md (used for tau_e)."""
    nv = meas.shape[1]
    per = np.zeros((steps, nv))
    for t in range(K, steps):
        cand = meas[t] >= r_min
        e = np.abs(fc[t - K] - meas[t])
        per[t] = (e * cand).sum(axis=1) / np.maximum(cand.sum(axis=1), 1)
    cs = np.cumsum(per, axis=0)
    roll = np.empty_like(per)
    for t in range(steps):
        lo = max(K, t - W_ERR + 1)
        roll[t] = (cs[t] - (cs[lo - 1] if lo > 0 else 0)) / max(t - lo + 1, 1) if t >= K else 0
    return roll


def select(provisional: bool = False):
    rows, res = [], {}
    for a in ARCHS:
        p = DATA / f"result-{a}.json"
        if not p.exists():
            if provisional:
                continue
            raise SystemExit(f"missing {p}")
        res[a] = json.loads(p.read_text())
        for split in ("val", "test"):
            r = res[a][split]
            rows.append({"arch": a, "split": split, "params": res[a]["params"],
                         "epochs_run": res[a]["epochs_run"],
                         "train_seconds": res[a]["train_seconds"],
                         "MAE_dB": r["model"]["MAE_dB"], "RMSE_dB": r["model"]["RMSE_dB"],
                         "R2": r["model"]["R2"], "changed_MAE_dB": r["model"]["changed_MAE_dB"],
                         "skill_vs_persistence": r["skill_vs_persistence"],
                         "persistence_MAE_dB": r["persistence"]["MAE_dB"],
                         "persistence_RMSE_dB": r["persistence"]["RMSE_dB"],
                         "persistence_changed_MAE_dB": r["persistence"]["changed_MAE_dB"],
                         "linear_MAE_dB": r["linear"]["MAE_dB"],
                         "linear_RMSE_dB": r["linear"]["RMSE_dB"],
                         "changed_frac": r["model"]["changed_frac"]})
    import pandas as pd
    pd.DataFrame(rows).to_csv(DATA / "accuracy.csv", index=False)
    chosen = min(res, key=lambda a: res[a]["val_mse_norm"])
    cfg = campaign_config()
    fc = TwinForecaster(chosen)
    errs = []
    for s in SPLITS["val"]:
        sc = Scenario(cfg, s)
        e = realised_error(fc.predict(sc), sc.meas, cfg.r_min_dbm, cfg.steps_per_episode)
        errs.append(e[W_ERR + K:].ravel())
    tau = float(np.percentile(np.concatenate(errs), 95))
    sel = {"selected_arch": chosen, "selection": "lowest validation MSE (normalised delta)",
           "val_mse_norm": {a: res[a]["val_mse_norm"] for a in res},
           "complete": len(res) == len(ARCHS),
           "tau_e_db": tau, "tau_rule": "95th percentile of e_t, clean forecasts, val scenarios",
           "W_steps": W_ERR, "K_steps": K, "H_steps": H}
    (DATA / "selection.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps(sel, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "select"])
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--provisional", action="store_true")
    a = ap.parse_args()
    if a.cmd == "data":
        build_data()
    elif a.cmd == "train":
        train(a.arch, a.threads)
    else:
        torch.set_num_threads(a.threads)
        select(a.provisional)


if __name__ == "__main__":
    main()
