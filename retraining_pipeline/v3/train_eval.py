"""
v3/train_eval.py
================
Train one architecture with a 5-output head (k = 1..5 s) on the v3 data and evaluate it in dB.
Training configuration is ../config.py, unchanged (see PREREGISTRATION.md).

Run: python train_eval.py --arch CNNLSTM --target abs|delta [--threads 4]
Writes results/{arch}_{target}.json and checkpoints/{arch}_{target}.pth.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C                                         # noqa: E402
from architectures import build_model                      # noqa: E402

DATA_DIR = os.path.join(HERE, "data")
RES_DIR = os.path.join(HERE, "results")
CK_DIR = os.path.join(HERE, "checkpoints")
RSRP_LO, RSRP_SPAN, K = -140.0, 96.0, 5


def load(name, target):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return d["X"], d[f"y_{target}"], d


def to_db(pred, d, target):
    if target == "abs":
        return pred * RSRP_SPAN + RSRP_LO
    return d["r_last"][:, None] + pred * RSRP_SPAN


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out.append(model(torch.from_numpy(X[i:i + 4096])).numpy())
    return np.concatenate(out)


def per_k_metrics(pred_db, d):
    rf, rl = d["r_future"], d["r_last"]
    res = []
    for k in range(K):
        t, p = rf[:, k], pred_db[:, k]
        e = t - p
        pe = t - rl
        changed = t != rl
        res.append({
            "k": k + 1,
            "MAE_dB": float(np.abs(e).mean()), "RMSE_dB": float(np.sqrt((e ** 2).mean())),
            "R2": float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum()),
            "skill_vs_persistence": float(1 - (e ** 2).mean() / (pe ** 2).mean()),
            "changed_frac": float(changed.mean()),
            "changed_MAE_dB": float(np.abs(e[changed]).mean()),
            "changed_persistence_MAE_dB": float(np.abs(pe[changed]).mean()),
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--target", choices=["abs", "delta"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(CK_DIR, exist_ok=True)

    Xtr, ytr, _ = load("train", args.target)
    Xva, yva, dva = load("val", args.target)
    Xte, _, dte = load("test", args.target)
    model = build_model(args.arch, C.INPUT_SIZE, K)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=C.LR_PATIENCE,
                                                       factor=C.LR_FACTOR)
    crit = nn.MSELoss()
    gen = torch.Generator().manual_seed(C.SEED)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True, generator=gen)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=1024, shuffle=False)
    best, best_state, patience, epochs = float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        epochs = epoch + 1
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([crit(model(xb), yb).item() for xb, yb in va]))
        sched.step(vloss)
        if vloss < best:
            best, patience = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 5 == 0:
            print(f"[{args.arch}/{args.target}] epoch {epochs} val_mse={vloss:.6f} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        if patience >= C.EARLY_STOP_PATIENCE:
            break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(CK_DIR, f"{args.arch}_{args.target}.pth"))
    pva, pte = to_db(predict(model, Xva), dva, args.target), to_db(predict(model, Xte), dte, args.target)
    out = {"arch": args.arch, "target": args.target, "params": params, "epochs_run": epochs,
           "train_seconds": round(time.perf_counter() - t0, 1),
           "val_mse_db_mean_over_k": float(np.mean((pva - dva["r_future"]) ** 2)),
           "val": per_k_metrics(pva, dva), "test": per_k_metrics(pte, dte)}
    with open(os.path.join(RES_DIR, f"{args.arch}_{args.target}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"arch": args.arch, "target": args.target, "epochs": epochs,
                      "test_MAE_k1..5": [round(r["MAE_dB"], 3) for r in out["test"]],
                      "skill": [round(r["skill_vs_persistence"], 3) for r in out["test"]]}))


if __name__ == "__main__":
    main()
