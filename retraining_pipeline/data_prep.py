"""
data_prep.py
============
Turn the raw public UCC/Cork 5G traces (Raca et al., "Beyond Throughput: a 4G & 5G
Cellular Dataset for Mobile Application Performance", MMSys'20) into model-ready
sequences for the six-architecture comparison.

What this script does:
  1. Loads every raw trace CSV in config.RAW_DATA_DIR (one file == one drive/trace).
  2. Engineers the 12 features the repo's inference contract expects
     (config.LSTM_FEATURES), from the raw UCC columns.
  3. Splits BY TRACE (not by row) into train/val/test — prevents temporal leakage.
  4. Fits ONE MinMaxScaler on the 11 non-RSRP features using TRAIN traces only;
     RSRP is normalized by the FIXED affine map (rsrp+120)/60 so the agent's
     denorm (r = 60*r_norm - 120) stays exactly valid and errors come out in dB.
  5. Builds (N, H=15, 12) input windows -> (N, 1) next-step RSRP targets.
  6. Saves processed tensors (.npz per split) + the single scaler artifact.

NOTHING here trains a model. It is safe to run, but it DOES require the dataset
to be present (see README "Getting the data"). Run:  python data_prep.py

------------------------------------------------------------------------------
HONESTY NOTE — feature reconstructibility (read before trusting the outputs):
The repo's 12 features are SIMULATION-engineered. From raw UCC traces:
  * Directly available / cleanly derivable:
      Longitude, Latitude, Speed, RSRP, Hour_sin, Hour_cos,
      Heading            (bearing between consecutive GPS points),
      Speed_normalized   (Speed / SPEED_MAX_KMH),
      RSRP_trend         (short finite-difference slope of RSRP history).
  * Requires cell geometry the raw traces don't always carry:
      Distance_serving, Distance_normalized
      -> UCC logs the serving CellID but not a canonical cell (lat,lon). We
         estimate each cell's position as the centroid of the GPS fixes recorded
         while camped on it (see _estimate_cell_positions). This is an APPROX;
         document it in the thesis. If a cell-site database is available, plug it
         into _estimate_cell_positions instead.
  * Event-derived:
      Handover_recent    (1 if CellID changed within HANDOVER_RECENT_WINDOW rows,
                          else decays to 0) — derived from CellID transitions.
These approximations are the price of the repo hardcoding sim-specific features
into the predictor contract. They do not affect RSRP units, only the auxiliary
inputs. Flagged so the chapter can state it rather than have the jury find it.
------------------------------------------------------------------------------
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

import config as C

SPEED_MAX_KMH = 120.0            # for Speed_normalized
HANDOVER_RECENT_WINDOW = 15      # rows over which a HO stays "recent"
DIST_NORM_MAX_M = 2000.0         # for Distance_normalized (grid diagonal-ish)

# Common column aliases across UCC dataset variants. Adjust to your CSV headers.
COLUMN_ALIASES = {
    "Longitude": ["Longitude", "longitude", "Lon", "lon", "GPS_Longitude"],
    "Latitude":  ["Latitude", "latitude", "Lat", "lat", "GPS_Latitude"],
    "Speed":     ["Speed", "speed", "Speed_kmh"],
    "RSRP":      ["RSRP", "rsrp", "NRxRSRP"],
    "CellID":    ["CellID", "cellid", "Cell_ID", "PCI", "ServingCell"],
    "Timestamp": ["Timestamp", "timestamp", "Time", "time", "datetime"],
}


def _resolve_columns(df):
    """Map a trace's actual headers to the canonical names we use downstream."""
    resolved = {}
    lower = {c.lower(): c for c in df.columns}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                resolved[canon] = a
                break
            if a.lower() in lower:
                resolved[canon] = lower[a.lower()]
                break
    missing = [k for k in ("Longitude", "Latitude", "Speed", "RSRP") if k not in resolved]
    if missing:
        raise ValueError(f"Trace missing required columns {missing}; "
                         f"headers were {list(df.columns)}")
    return resolved


def _bearing(lat1, lon1, lat2, lon2):
    """Compass heading (deg) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _estimate_cell_positions(df):
    """Approx each CellID's location as the centroid of GPS fixes on that cell.
    Replace with a real cell-site DB if you have one."""
    if "CellID" not in df.columns:
        return {}
    return {cid: (g["Latitude"].mean(), g["Longitude"].mean())
            for cid, g in df.groupby("CellID")}


def engineer_features(df):
    """Produce the 12 features (+ target) for a single trace, in order."""
    cols = _resolve_columns(df)
    # UCC traces occasionally carry '-' placeholders in some columns; coerce the
    # required numeric ones and drop rows that are unusable for the predictor.
    df = df.copy()
    for canon in ("Longitude", "Latitude", "Speed", "RSRP"):
        df[cols[canon]] = pd.to_numeric(df[cols[canon]], errors="coerce")
    df = df.dropna(subset=[cols["Longitude"], cols["Latitude"],
                           cols["Speed"], cols["RSRP"]]).reset_index(drop=True)

    out = pd.DataFrame()
    out["Longitude"] = df[cols["Longitude"]].astype(float)
    out["Latitude"]  = df[cols["Latitude"]].astype(float)
    out["Speed"]     = df[cols["Speed"]].astype(float)
    out["RSRP"]      = df[cols["RSRP"]].astype(float).clip(C.RSRP_MIN_DBM, C.RSRP_MAX_DBM)

    # Heading from consecutive GPS points
    lat, lon = out["Latitude"].values, out["Longitude"].values
    heading = np.zeros(len(out))
    heading[1:] = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    out["Heading"] = heading

    # Hour of day -> cyclic encoding.
    # UCC/Raca timestamps look like "2019.12.16_14.23.32"; try that format first,
    # then fall back to a generic parse.
    if "Timestamp" in cols:
        raw_ts = df[cols["Timestamp"]]
        ts = pd.to_datetime(raw_ts, format="%Y.%m.%d_%H.%M.%S", errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(raw_ts, errors="coerce")
        hour = ts.dt.hour.fillna(0).values
    else:
        hour = np.zeros(len(out))
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Handover_recent from CellID transitions
    if "CellID" in cols:
        cid = df[cols["CellID"]].values
        changed = np.zeros(len(out))
        changed[1:] = (cid[1:] != cid[:-1]).astype(float)
        recent = np.zeros(len(out))
        countdown = 0
        for i in range(len(out)):
            if changed[i]:
                countdown = HANDOVER_RECENT_WINDOW
            recent[i] = 1.0 if countdown > 0 else 0.0
            countdown = max(0, countdown - 1)
        out["Handover_recent"] = recent
    else:
        out["Handover_recent"] = 0.0

    # RSRP_trend: short finite-difference slope (normalized by span)
    rsrp = out["RSRP"].values
    trend = np.zeros(len(out))
    trend[1:] = (rsrp[1:] - rsrp[:-1]) / (C.RSRP_MAX_DBM - C.RSRP_MIN_DBM)
    out["RSRP_trend"] = trend

    out["Speed_normalized"] = (out["Speed"] / SPEED_MAX_KMH).clip(0, 1)

    # Distance to serving cell (approx from estimated cell centroids)
    if "CellID" in cols:
        cellpos = _estimate_cell_positions(
            df.assign(Latitude=out["Latitude"], Longitude=out["Longitude"],
                      CellID=df[cols["CellID"]]))
        cid = df[cols["CellID"]].values
        dist = np.array([
            _haversine_m(out["Latitude"].values[i], out["Longitude"].values[i],
                         *cellpos.get(cid[i], (out["Latitude"].values[i],
                                               out["Longitude"].values[i])))
            for i in range(len(out))
        ])
    else:
        dist = np.zeros(len(out))
    out["Distance_serving"] = dist
    out["Distance_normalized"] = np.clip(dist / DIST_NORM_MAX_M, 0, 1)

    # order exactly as the contract expects
    return out[C.LSTM_FEATURES].astype(float)


def _rsrp_affine_norm(x):
    return np.clip((x - C.RSRP_MIN_DBM) / (C.RSRP_MAX_DBM - C.RSRP_MIN_DBM), 0.0, 1.0)


def make_sequences(arr, targets):
    """(T, 12) -> (N, H, 12) windows, next-step target -> (N, 1)."""
    H = C.SEQUENCE_LENGTH
    X, y = [], []
    for i in range(H, len(arr)):
        X.append(arr[i - H:i])
        y.append(targets[i])
    if not X:
        return np.empty((0, H, C.INPUT_SIZE)), np.empty((0, 1))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)


def split_traces(trace_files):
    """Split the LIST OF TRACE FILES (not rows) into train/val/test."""
    rng = np.random.default_rng(C.SPLIT_SEED)
    files = list(trace_files)
    rng.shuffle(files)
    n = len(files)
    n_tr = int(round(C.TRAIN_FRAC * n))
    n_va = int(round(C.VAL_FRAC * n))
    return {"train": files[:n_tr],
            "val":   files[n_tr:n_tr + n_va],
            "test":  files[n_tr + n_va:]}


def main():
    os.makedirs(C.PROC_DATA_DIR, exist_ok=True)
    trace_files = sorted(glob.glob(os.path.join(C.RAW_DATA_DIR, "*.csv")))
    if not trace_files:
        raise SystemExit(
            f"No trace CSVs found in {C.RAW_DATA_DIR}. "
            f"See README 'Getting the data'.")
    print(f"Found {len(trace_files)} trace files.")

    splits = split_traces(trace_files)
    for name, files in splits.items():
        print(f"  {name}: {len(files)} traces")

    # 1) engineer features per trace, keep per-trace to avoid cross-trace windows
    engineered = {}   # path -> DataFrame(12 cols)
    for f in trace_files:
        try:
            df = pd.read_csv(f)
            engineered[f] = engineer_features(df)
        except Exception as e:
            print(f"  ! skipping {os.path.basename(f)}: {e}")

    # 2) fit the feature scaler on TRAIN traces only (non-RSRP cols); RSRP fixed
    non_rsrp_cols = [c for c in C.LSTM_FEATURES if c != C.TARGET_FEATURE]
    train_stack = pd.concat([engineered[f][non_rsrp_cols]
                             for f in splits["train"] if f in engineered],
                            axis=0)
    feat_scaler = MinMaxScaler().fit(train_stack.values)

    # persist ONE artifact describing the full normalization
    joblib.dump({
        "feature_scaler": feat_scaler,
        "non_rsrp_cols": non_rsrp_cols,
        "rsrp_min_dbm": C.RSRP_MIN_DBM,
        "rsrp_max_dbm": C.RSRP_MAX_DBM,
        "feature_order": C.LSTM_FEATURES,
        "sequence_length": C.SEQUENCE_LENGTH,
        "note": "RSRP normalized by fixed affine (rsrp+120)/60; "
                "others by MinMaxScaler fit on train traces.",
    }, C.SCALER_PATH)
    print(f"Saved scaler -> {C.SCALER_PATH}")

    # 3) transform + window each split (windows built WITHIN a trace only).
    # A per-window integer trace id (`tid`) is saved alongside X/y so that
    # multi-step evaluation (evaluate.py) can refuse to read ground truth
    # across a trace boundary — consecutive windows in the concatenated arrays
    # are NOT guaranteed to be temporally consecutive at trace joins.
    trace_index = {f: i for i, f in enumerate(sorted(engineered))}
    for name, files in splits.items():
        Xs, ys, tids = [], [], []
        for f in files:
            if f not in engineered:
                continue
            e = engineered[f].copy()
            e[non_rsrp_cols] = feat_scaler.transform(e[non_rsrp_cols].values)
            e[C.TARGET_FEATURE] = _rsrp_affine_norm(e[C.TARGET_FEATURE].values)
            arr = e[C.LSTM_FEATURES].values.astype(np.float32)
            tgt = e[C.TARGET_FEATURE].values.astype(np.float32)   # normalized RSRP
            X, y = make_sequences(arr, tgt)
            if len(X):
                Xs.append(X)
                ys.append(y)
                tids.append(np.full(len(X), trace_index[f], dtype=np.int32))
        X = np.concatenate(Xs) if Xs else np.empty((0, C.SEQUENCE_LENGTH, C.INPUT_SIZE), np.float32)
        y = np.concatenate(ys) if ys else np.empty((0, 1), np.float32)
        tid = np.concatenate(tids) if tids else np.empty((0,), np.int32)
        out = os.path.join(C.PROC_DATA_DIR, f"{name}.npz")
        np.savez_compressed(out, X=X, y=y, tid=tid)
        print(f"  wrote {out}: X={X.shape}, y={y.shape}, traces={len(np.unique(tid))}")

    print("Done. Next: python train.py")


if __name__ == "__main__":
    main()
