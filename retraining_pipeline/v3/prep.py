"""
v3/prep.py
==========
Phase-1a data preparation (see PREREGISTRATION.md): LTE/5G rows with valid RSRP only,
contiguous segments, no RSRP clipping, 5-step targets. Reuses the geometry helpers and the
by-trace split of ../data_prep.py so the split is identical to the earlier pipeline.

Writes data/{train,val,test}.npz with
  X (N, 15, 12)      normalized inputs
  y_abs (N, 5)       (r_{t+k} + 140) / 96
  y_delta (N, 5)     (r_{t+k} - r_t) / 96
  r_last, r_prev (N,), r_future (N, 5)   in dBm, for the naive baselines
  tid, seg (N,)      trace and segment ids
and data/scaler.joblib.

Run: python prep.py
"""

import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C                                                          # noqa: E402
from data_prep import (_bearing, _haversine_m, split_traces,               # noqa: E402
                       SPEED_MAX_KMH, HANDOVER_RECENT_WINDOW, DIST_NORM_MAX_M)

DATA_DIR = os.path.join(HERE, "data")
RSRP_LO, RSRP_HI = -140.0, -44.0
RSRP_SPAN = RSRP_HI - RSRP_LO                 # 96 dB
KEEP_MODES = {"LTE", "5G"}
MAX_GAP_S = 2.0
K = 5
H = C.SEQUENCE_LENGTH


def norm_rsrp(r):
    return (r - RSRP_LO) / RSRP_SPAN


def engineer(path):
    """Filtered rows of one trace -> (features DataFrame in contract order, rsrp dBm, segment id)."""
    df = pd.read_csv(path)
    for c in ("Longitude", "Latitude", "Speed", "RSRP"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    df = df.dropna(subset=["Longitude", "Latitude", "Speed", "RSRP", "ts"])
    keep = df["NetworkMode"].isin(KEEP_MODES) & df["RSRP"].between(RSRP_LO, RSRP_HI)
    pos = np.arange(len(df))
    df = df[keep.values].copy()
    pos = pos[keep.values]
    if len(df) == 0:
        return None
    gap = df["ts"].diff().dt.total_seconds().fillna(0).values
    jump = np.r_[False, np.diff(pos) > 1]
    seg = np.cumsum((gap > MAX_GAP_S) | jump)
    start = np.r_[True, seg[1:] != seg[:-1]]

    lat, lon = df["Latitude"].values, df["Longitude"].values
    out = pd.DataFrame(index=range(len(df)))
    out["Longitude"], out["Latitude"] = lon, lat
    out["Speed"] = df["Speed"].values
    rsrp = df["RSRP"].values.astype(float)
    out["RSRP"] = rsrp
    cells = {cid: (g["Latitude"].mean(), g["Longitude"].mean()) for cid, g in df.groupby("CellID")}
    cid = df["CellID"].values
    clat = np.array([cells[c][0] for c in cid])
    clon = np.array([cells[c][1] for c in cid])
    dist = _haversine_m(lat, lon, clat, clon)
    out["Distance_serving"] = dist
    heading = np.zeros(len(df))
    heading[1:] = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    heading[start] = np.r_[heading[1:], 0][start]          # no bearing across a segment start
    out["Heading"] = heading
    hour = df["ts"].dt.hour.values + df["ts"].dt.minute.values / 60.0
    out["Hour_sin"], out["Hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    recent, countdown = np.zeros(len(df)), 0
    for i in range(len(df)):
        if start[i]:
            countdown = 0
        elif cid[i] != cid[i - 1]:
            countdown = HANDOVER_RECENT_WINDOW
        recent[i] = 1.0 if countdown > 0 else 0.0
        countdown = max(0, countdown - 1)
    out["Handover_recent"] = recent
    trend = np.zeros(len(df))
    trend[1:] = (rsrp[1:] - rsrp[:-1]) / RSRP_SPAN
    trend[start] = 0.0
    out["RSRP_trend"] = trend
    out["Speed_normalized"] = np.clip(out["Speed"].values / SPEED_MAX_KMH, 0, 1)
    out["Distance_normalized"] = np.clip(dist / DIST_NORM_MAX_M, 0, 1)
    return out[C.LSTM_FEATURES].astype(float), rsrp, seg


def windows(arr, rsrp, seg):
    X, ya, yd, rl, rp, rf, sg = [], [], [], [], [], [], []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        if len(idx) < H + K:
            continue
        a, r = arr[idx], rsrp[idx]
        for i in range(H, len(idx) - K + 1):
            X.append(a[i - H:i])
            fut = r[i:i + K]
            ya.append(norm_rsrp(fut))
            yd.append((fut - r[i - 1]) / RSRP_SPAN)
            rl.append(r[i - 1])
            rp.append(r[i - 2])
            rf.append(fut)
            sg.append(s)
    f32 = lambda v: np.asarray(v, dtype=np.float32)
    return f32(X), f32(ya), f32(yd), f32(rl), f32(rp), f32(rf), np.asarray(sg, np.int32)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(C.RAW_DATA_DIR, "*.csv")))
    splits = split_traces(files)
    eng = {f: engineer(f) for f in files}
    eng = {f: v for f, v in eng.items() if v is not None}
    non_rsrp = [c for c in C.LSTM_FEATURES if c != C.TARGET_FEATURE]
    scaler = MinMaxScaler().fit(pd.concat([eng[f][0][non_rsrp] for f in splits["train"]
                                           if f in eng]).values)
    joblib.dump({"feature_scaler": scaler, "non_rsrp_cols": non_rsrp, "rsrp_norm": "(r+140)/96",
                 "feature_order": C.LSTM_FEATURES, "H": H, "K": K}, os.path.join(DATA_DIR, "scaler.joblib"))
    tindex = {f: i for i, f in enumerate(files)}
    stats = {}
    for name, fl in splits.items():
        parts = []
        for f in fl:
            if f not in eng:
                continue
            feat, rsrp, seg = eng[f]
            feat = feat.copy()
            feat[non_rsrp] = scaler.transform(feat[non_rsrp].values)
            feat["RSRP"] = norm_rsrp(feat["RSRP"].values)
            w = windows(feat[C.LSTM_FEATURES].values.astype(np.float32), rsrp, seg)
            if len(w[0]):
                parts.append(w + (np.full(len(w[0]), tindex[f], np.int32),))
        cat = [np.concatenate([p[j] for p in parts]) for j in range(8)]
        keys = ["X", "y_abs", "y_delta", "r_last", "r_prev", "r_future", "seg", "tid"]
        np.savez_compressed(os.path.join(DATA_DIR, f"{name}.npz"), **dict(zip(keys, cat)))
        stats[name] = {"traces": len(fl), "windows": int(len(cat[0]))}
        print(name, stats[name])
    rows = sum(len(eng[f][1]) for f in eng)
    print("kept rows (LTE/5G, valid RSRP):", rows)


if __name__ == "__main__":
    main()
