"""v3 simulated network ("twin") for the Phase-0 go/no-go test.

Changes from v2 (see ../PREREGISTRATION.md):
  * the radio channel contains no learned model: TR 38.901 UMa/UMi-SC path loss and
    LOS probability, spatially consistent LOS state, spatially correlated shadowing;
  * RSRP per resource element, residual measurement fading, L3 filtering;
  * SINR-based handover failure (HOF), radio link failure (RLF) and outage;
  * Manhattan-grid mobility (random walk kept for later sweeps).

The decision agent, its hyperparameters, the reward and the constraint-layer rules are
imported unchanged from ch5-outputs-v2/scripts/common.py.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.special import ndtr

OUTPUT_ROOT = Path(__file__).resolve().parents[1]
V2_COMMON = OUTPUT_ROOT.parent / "ch5-outputs-v2" / "scripts" / "common.py"

_spec = importlib.util.spec_from_file_location("v2common", V2_COMMON)
v2 = importlib.util.module_from_spec(_spec)
sys.modules["v2common"] = v2      # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(v2)

DQNAgent = v2.DQNAgent
shaped_reward = v2.shaped_reward
make_cells = v2.make_cells
set_global_seed = v2.set_global_seed
signal_quality_numeric = v2.signal_quality_numeric
runtime_hardware = v2.runtime_hardware

SPEED_OF_LIGHT = 3.0e8


@dataclass(frozen=True)
class TwinConfig(v2.PinnedConfig):
    # radio (sources: ../config/parameter-sources.md)
    carrier_ghz: float = 3.5
    n_rb: int = 273                       # 100 MHz at 30 kHz SCS
    scs_hz: float = 30e3
    tx_power_dbm: tuple = (49.0, 37.0, 30.0)   # macro, micro, pico
    tx_offset_db: float = 0.0             # calibration knob (pre-registered)
    antenna_gain_dbi: tuple = (8.0, 5.0, 5.0)
    h_bs_m: tuple = (25.0, 10.0, 10.0)
    h_ut_m: float = 1.5
    noise_figure_db: float = 9.0
    sf_std_db: tuple = ((4.0, 6.0), (4.0, 7.82))       # (LOS, NLOS) for UMa, UMi
    sf_dcorr_m: tuple = ((37.0, 50.0), (10.0, 13.0))   # (LOS, NLOS) for UMa, UMi
    los_dcorr_m: tuple = (50.0, 50.0)                  # UMa, UMi
    residual_fading_db: float = 1.0
    l3_k: int = 4
    # link-level thresholds
    q_out_db: float = -8.0
    q_in_db: float = -6.0
    t310_steps: int = 5                   # 1 s at 0.2 s
    t_store_s: float = 5.0                # MRO classification window
    pp_mts_s: float = 1.0
    pp_secondary_s: float = 10.0
    r_min_dbm: float = -125.0             # measurement-report threshold for candidates
    # mobility
    mobility: str = "manhattan"
    block_m: float = 250.0
    speed_min_kmh: float = 60.0
    speed_max_kmh: float = 100.0
    # agent-side
    recent_window_s: float = 60.0         # s12
    rsrp_norm_offset: float = 140.0       # (r + 140) / 80
    rsrp_norm_scale: float = 80.0
    horizon_pad: int = 10                 # extra precomputed steps for oracle look-ahead
    evaluation_scenarios: int = 20
    calibration_seeds: tuple = (2000, 2001, 2002, 2003, 2004)
    # Phase 0b switches (defaults reproduce Phase 0)
    channel_version: int = 1              # 2: smooth LOS variable, one shadowing process per link
    failure_aware_reward: bool = False    # add outage / RLF / HOF penalties to the v2 reward
    outage_penalty: float = -1.0          # per step with serving SINR < Q_out (next step)
    rlf_penalty: float = -15.0
    hof_penalty: float = -15.0
    reestablish_is_change: bool = False   # RLF re-establishment resets the time since change
    # Phase 0c switches (PREREGISTRATION-0c.md; defaults reproduce Phase 0 / 0b)
    reliability_override: bool = False    # interval + blacklist don't block when SINR_meas is low
    override_margin_db: float = 4.0       # override if serving SINR_meas < Q_out + margin
    admission_check: bool = False         # execute only if target SINR_meas >= Q_in
    # Phase 2 amendment (PREREGISTRATION-2-AMENDMENT.md; default reproduces Phases 0-0c)
    reward_mode: str = "v2"               # "kpi": outage / RLF / HOF / HO cost / ping-pong only
    ho_cost: float = 0.1
    pp_penalty: float = 1.5
    # Study 3 (PREREGISTRATION-3.md, post hoc): forecast-based agents use measurements in danger
    danger_uses_measurement: bool = False


def cell_kind_index(cell) -> int:
    return {"macro": 0, "micro": 1, "pico": 2}[cell.cell_type]


# --------------------------------------------------------------------------- channel

def pathloss_db(d2d: np.ndarray, los: np.ndarray, umi: np.ndarray, h_bs: np.ndarray,
                h_ut: float, fc_ghz: float) -> np.ndarray:
    """TR 38.901 Table 7.4.1-1, UMa (umi=False) and UMi-Street Canyon (umi=True)."""
    d2d = np.maximum(d2d, 10.0)
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    h_e = 1.0
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_ghz * 1e9 / SPEED_OF_LIGHT
    lf = 20 * np.log10(fc_ghz)
    # UMa
    uma_1 = 28.0 + 22 * np.log10(d3d) + lf
    uma_2 = 28.0 + 40 * np.log10(d3d) + lf - 9 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    uma_los = np.where(d2d <= d_bp, uma_1, uma_2)
    uma_nlos = np.maximum(uma_los, 13.54 + 39.08 * np.log10(d3d) + lf - 0.6 * (h_ut - 1.5))
    # UMi-Street Canyon
    umi_1 = 32.4 + 21 * np.log10(d3d) + lf
    umi_2 = 32.4 + 40 * np.log10(d3d) + lf - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    umi_los = np.where(d2d <= d_bp, umi_1, umi_2)
    umi_nlos = np.maximum(umi_los, 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc_ghz)
                          - 0.3 * (h_ut - 1.5))
    pl_los = np.where(umi, umi_los, uma_los)
    pl_nlos = np.where(umi, umi_nlos, uma_nlos)
    return np.where(los, pl_los, pl_nlos)


def los_probability(d2d: np.ndarray, umi: np.ndarray) -> np.ndarray:
    """TR 38.901 Table 7.4.2-1 (h_UT <= 13 m)."""
    d = np.maximum(d2d, 1e-6)
    uma = np.where(d <= 18, 1.0, 18 / d + np.exp(-d / 63) * (1 - 18 / d))
    umi_p = np.where(d <= 18, 1.0, 18 / d + np.exp(-d / 36) * (1 - 18 / d))
    return np.where(umi, umi_p, uma)


# -------------------------------------------------------------------------- mobility

_DIRS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)   # E, N, W, S


def manhattan_trajectories(cfg: TwinConfig, rng: np.random.Generator, steps: int):
    """Vehicles on a street grid; at intersections: straight 0.5, left 0.25, right 0.25."""
    w, h, b = cfg.grid_width_m, cfg.grid_height_m, cfg.block_m
    nv = cfg.n_vehicles
    xs, ys = np.arange(0, w + 1e-6, b), np.arange(0, h + 1e-6, b)
    x, y = np.empty(nv), np.empty(nv)
    d = np.empty(nv, dtype=int)
    for i in range(nv):
        if rng.random() < 0.5:                       # on a horizontal street
            y[i] = rng.choice(ys)
            x[i] = rng.uniform(0, w)
            d[i] = rng.choice([0, 2])
        else:
            x[i] = rng.choice(xs)
            y[i] = rng.uniform(0, h)
            d[i] = rng.choice([1, 3])
    speed = rng.uniform(cfg.speed_min_kmh, cfg.speed_max_kmh, nv)
    out_x, out_y = np.empty((steps, nv)), np.empty((steps, nv))
    out_v, out_hd = np.empty((steps, nv)), np.empty((steps, nv))

    def valid(px, py, k):
        nx, ny = px + _DIRS[k, 0] * 1e-3, py + _DIRS[k, 1] * 1e-3
        return -1e-6 <= nx <= w + 1e-6 and -1e-6 <= ny <= h + 1e-6

    for t in range(steps):
        speed = np.clip(speed + rng.uniform(-2, 2, nv), cfg.speed_min_kmh, cfg.speed_max_kmh)
        for i in range(nv):
            remaining = speed[i] * cfg.dt_s / 3.6
            while remaining > 1e-9:
                dx, dy = _DIRS[d[i]]
                if dx != 0:   # distance to next vertical street in the direction of travel
                    grid = xs
                    pos = x[i]
                else:
                    grid = ys
                    pos = y[i]
                sgn = dx if dx != 0 else dy
                ahead = grid[(grid - pos) * sgn > 1e-9]
                if len(ahead) == 0:
                    nxt_dist = np.inf
                else:
                    nxt = ahead.min() if sgn > 0 else ahead.max()
                    nxt_dist = abs(nxt - pos)
                if remaining < nxt_dist:
                    x[i] += dx * remaining
                    y[i] += dy * remaining
                    remaining = 0.0
                else:
                    x[i] += dx * nxt_dist
                    y[i] += dy * nxt_dist
                    remaining -= nxt_dist
                    opts = [(d[i], 0.5), ((d[i] + 1) % 4, 0.25), ((d[i] + 3) % 4, 0.25)]
                    opts = [(k, p) for k, p in opts if valid(x[i], y[i], k)]
                    if not opts:
                        opts = [((d[i] + 2) % 4, 1.0)]
                    ks, ps = zip(*opts)
                    ps = np.asarray(ps) / sum(ps)
                    d[i] = ks[int(rng.choice(len(ks), p=ps))]
        out_x[t], out_y[t], out_v[t] = x, y, speed
        out_hd[t] = np.degrees(np.arctan2(_DIRS[d, 1], _DIRS[d, 0])) % 360
    return out_x, out_y, out_v, out_hd


def random_walk_trajectories(cfg: TwinConfig, rng: np.random.Generator, steps: int):
    """v2 mobility: heading/speed random walk with boundary reflection."""
    nv = cfg.n_vehicles
    x = rng.uniform(0, cfg.grid_width_m, nv)
    y = rng.uniform(0, cfg.grid_height_m, nv)
    speed = rng.uniform(cfg.speed_min_kmh, cfg.speed_max_kmh, nv)
    heading = rng.uniform(0, 360, nv)
    out = [np.empty((steps, nv)) for _ in range(4)]
    for t in range(steps):
        heading = (heading + rng.uniform(-10, 10, nv)) % 360
        speed = np.clip(speed + rng.uniform(-5, 5, nv), cfg.speed_min_kmh, cfg.speed_max_kmh)
        step = speed * cfg.dt_s / 3.6
        nx = x + step * np.cos(np.radians(heading))
        ny = y + step * np.sin(np.radians(heading))
        hx = (nx < 0) | (nx > cfg.grid_width_m)
        hy = (ny < 0) | (ny > cfg.grid_height_m)
        heading[hx] = (180 - heading[hx]) % 360
        heading[hy] = (-heading[hy]) % 360
        x, y = np.clip(nx, 0, cfg.grid_width_m), np.clip(ny, 0, cfg.grid_height_m)
        out[0][t], out[1][t], out[2][t], out[3][t] = x, y, speed, heading
    return tuple(out)


def random_waypoint_trajectories(cfg: TwinConfig, rng: np.random.Generator, steps: int):
    """Random Waypoint without pause: straight legs to uniform waypoints, new speed per leg."""
    nv, w, h = cfg.n_vehicles, cfg.grid_width_m, cfg.grid_height_m
    x, y = rng.uniform(0, w, nv), rng.uniform(0, h, nv)
    wx, wy = rng.uniform(0, w, nv), rng.uniform(0, h, nv)
    speed = rng.uniform(cfg.speed_min_kmh, cfg.speed_max_kmh, nv)
    out = [np.empty((steps, nv)) for _ in range(4)]
    for t in range(steps):
        for i in range(nv):
            remaining = speed[i] * cfg.dt_s / 3.6
            while remaining > 1e-9:
                dx, dy = wx[i] - x[i], wy[i] - y[i]
                d = math.hypot(dx, dy)
                if d <= remaining:
                    x[i], y[i] = wx[i], wy[i]
                    remaining -= d
                    wx[i], wy[i] = rng.uniform(0, w), rng.uniform(0, h)
                    speed[i] = rng.uniform(cfg.speed_min_kmh, cfg.speed_max_kmh)
                else:
                    x[i] += dx / d * remaining
                    y[i] += dy / d * remaining
                    remaining = 0.0
        heading = np.degrees(np.arctan2(wy - y, wx - x)) % 360
        out[0][t], out[1][t], out[2][t], out[3][t] = x, y, speed, heading
    return tuple(out)


def gauss_markov_trajectories(cfg: TwinConfig, rng: np.random.Generator, steps: int,
                              alpha: float = 0.75, update_s: float = 1.0,
                              sigma_heading: float = 20.0, sigma_speed: float = 5.0):
    """Gauss-Markov mobility (Camp et al. 2002): speed and heading are AR(1) around their
    means, updated every update_s; the mean heading turns towards the centre near the edges."""
    nv, w, h = cfg.n_vehicles, cfg.grid_width_m, cfg.grid_height_m
    x, y = rng.uniform(0, w, nv), rng.uniform(0, h, nv)
    mean_speed = (cfg.speed_min_kmh + cfg.speed_max_kmh) / 2
    speed = rng.uniform(cfg.speed_min_kmh, cfg.speed_max_kmh, nv)
    heading = rng.uniform(0, 360, nv)
    mean_heading = heading.copy()
    every = max(1, int(round(update_s / cfg.dt_s)))
    c = math.sqrt(1 - alpha ** 2)
    out = [np.empty((steps, nv)) for _ in range(4)]
    for t in range(steps):
        if t % every == 0 and t:
            margin = 150.0
            edge = (x < margin) | (x > w - margin) | (y < margin) | (y > h - margin)
            centre = np.degrees(np.arctan2(h / 2 - y, w / 2 - x)) % 360
            mean_heading = np.where(edge, centre, mean_heading)
            diff = (mean_heading - heading + 180) % 360 - 180
            heading = (heading + (1 - alpha) * diff + c * sigma_heading *
                       rng.normal(0, 1, nv)) % 360
            speed = np.clip(alpha * speed + (1 - alpha) * mean_speed +
                            c * sigma_speed * rng.normal(0, 1, nv),
                            cfg.speed_min_kmh, cfg.speed_max_kmh)
        step = speed * cfg.dt_s / 3.6
        x = np.clip(x + step * np.cos(np.radians(heading)), 0, w)
        y = np.clip(y + step * np.sin(np.radians(heading)), 0, h)
        out[0][t], out[1][t], out[2][t], out[3][t] = x, y, speed, heading
    return tuple(out)


MOBILITY = {"manhattan": manhattan_trajectories, "random_walk": random_walk_trajectories,
            "random_waypoint": random_waypoint_trajectories,
            "gauss_markov": gauss_markov_trajectories}


# -------------------------------------------------------------------------- scenario

class Scenario:
    """Policy-independent mobility, loads and radio channel for one episode (precomputed)."""

    def __init__(self, cfg: TwinConfig, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        cells = make_cells(cfg)
        self.cells = cells
        nv, nc = cfg.n_vehicles, len(cells)
        steps = cfg.steps_per_episode + cfg.horizon_pad
        self.steps = steps
        mobility = cfg.mobility if cfg.mobility in MOBILITY else "random_walk"
        x, y, v, hd = MOBILITY[mobility](cfg, rng, steps)
        self.x, self.y, self.speed, self.heading = x, y, v, hd

        kind = np.array([cell_kind_index(c) for c in cells])
        umi = kind > 0
        cx = np.array([c.x_m for c in cells])
        cy = np.array([c.y_m for c in cells])
        self.cells_x, self.cells_y = cx, cy
        h_bs = np.array(cfg.h_bs_m)[kind]
        p_tx = np.array(cfg.tx_power_dbm)[kind] + cfg.tx_offset_db
        g_tx = np.array(cfg.antenna_gain_dbi)[kind]
        re_db = 10 * math.log10(12 * cfg.n_rb)
        sf_std = np.array(cfg.sf_std_db)[umi.astype(int)]        # (nc, 2)
        sf_dc = np.array(cfg.sf_dcorr_m)[umi.astype(int)]        # (nc, 2)
        los_dc = np.array(cfg.los_dcorr_m)[umi.astype(int)]      # (nc,)

        dist = np.hypot(x[:, :, None] - cx[None, None, :], y[:, :, None] - cy[None, None, :])
        self.dist = dist
        moved = np.zeros((steps, nv))
        moved[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))

        los = np.empty((steps, nv, nc), bool)
        sf = np.empty((steps, nv, nc))
        p_los = los_probability(dist, umi[None, None, :])
        if cfg.channel_version == 1:
            # Phase 0: first-order LOS variable, independent LOS and NLOS shadowing processes
            g = rng.normal(0, 1, (nv, nc))
            z_los = rng.normal(0, 1, (nv, nc))
            z_nlos = rng.normal(0, 1, (nv, nc))
            for t in range(steps):
                if t:
                    m = moved[t][:, None]
                    for arr, dc in ((g, los_dc[None, :]), (z_los, sf_dc[None, :, 0]),
                                    (z_nlos, sf_dc[None, :, 1])):
                        rho = np.exp(-m / dc)
                        arr *= rho
                        arr += np.sqrt(1 - rho ** 2) * rng.normal(0, 1, (nv, nc))
                los[t] = ndtr(g) < p_los[t]
                sf[t] = np.where(los[t], sf_std[None, :, 0] * z_los, sf_std[None, :, 1] * z_nlos)
        else:
            # Phase 0b: smooth spatial processes (two cascaded AR(1) stages = Matern-3/2
            # correlation, 1/e at the TR 38.901 correlation distance), as obtained by the
            # grid-and-interpolate spatial-consistency procedure; one shadowing process per
            # link, scaled by the current LOS state (no jump at LOS changes)
            cascade_std = lambda r: np.sqrt((1 + r ** 2) / np.maximum(1 - r ** 2, 1e-12))
            m0 = (v[0] * cfg.dt_s / 3.6)[:, None]
            procs = []
            for dc in (los_dc[None, :] / 2.146, sf_dc[None, :, 1] / 2.146):
                r0 = np.exp(-m0 / dc)
                procs.append([dc, rng.normal(0, 1, (nv, nc)),
                              rng.normal(0, 1, (nv, nc)) * cascade_std(r0), r0])
            for t in range(steps):
                m = moved[t][:, None]
                out = []
                for p in procs:
                    dc, s1, s2, rho = p
                    if t:
                        rho = np.exp(-m / dc)
                        c = np.sqrt(1 - rho ** 2)
                        s1 = rho * s1 + c * rng.normal(0, 1, (nv, nc))
                        s2 = rho * s2 + c * s1
                        p[1], p[2], p[3] = s1, s2, rho
                    out.append(s2 / cascade_std(rho))
                g, z = out
                los[t] = ndtr(g) < p_los[t]
                sf[t] = np.where(los[t], sf_std[None, :, 0], sf_std[None, :, 1]) * z
        self.los = los
        pl = pathloss_db(dist, los, umi[None, None, :], h_bs[None, None, :], cfg.h_ut_m,
                         cfg.carrier_ghz)
        truth = (p_tx - re_db + g_tx)[None, None, :] - pl - sf
        self.truth = truth                                       # true RSRP per RE (dBm)

        # residual measurement fading + L3 filter (TS 38.331 5.5.3.2)
        l1 = truth + rng.normal(0, cfg.residual_fading_db, truth.shape)
        a = 1.0 / 2 ** (cfg.l3_k / 4)
        meas = np.empty_like(l1)
        meas[0] = l1[0]
        for t in range(1, steps):
            meas[t] = (1 - a) * meas[t - 1] + a * l1[t]
        self.meas = meas

        # cell loads (v2 process) -> interference activity factor
        loads = np.empty((steps, nc))
        cur = rng.integers(0, 11, nc).astype(float)
        for t in range(steps):
            if t % 50 == 0 and t:
                change = rng.integers(-2, 4, nc)
                active = rng.random(nc) < 0.1
                cur = np.clip(cur + change * active, 0, 30)
            loads[t] = cur
        self.loads = loads
        eta = 0.2 + 0.6 * loads / 30.0
        noise_dbm = -174 + 10 * math.log10(cfg.scs_hz) + cfg.noise_figure_db
        self.noise_dbm = noise_dbm
        self.sinr_true = self._sinr(truth, eta, noise_dbm)
        self.sinr_meas = self._sinr(meas, eta, noise_dbm)

    @staticmethod
    def _sinr(rsrp_dbm: np.ndarray, eta: np.ndarray, noise_dbm: float) -> np.ndarray:
        p = 10 ** (rsrp_dbm / 10)
        weighted = p * eta[:, None, :]
        interf = weighted.sum(axis=2, keepdims=True) - weighted
        return 10 * np.log10(p / (interf + 10 ** (noise_dbm / 10)))


# --------------------------------------------------------------------------- runtime

ARM_REACTIVE = "Reactive DQN"
ARM_ORACLE5 = "Oracle DQN k=5"
ARM_ORACLE10 = "Oracle DQN k=10"
ARM_RULES = "Rules only"
ARM_A3 = "Static A3"
ARM_A3T = "Tuned A3"
LEARNING_ARMS = {ARM_REACTIVE: 0, ARM_ORACLE5: 5, ARM_ORACLE10: 10}


class Runtime:
    """Per-arm connection state, constraint layer, reward and KPI counters."""

    def __init__(self, cfg: TwinConfig, sc: Scenario):
        self.cfg, self.sc = cfg, sc
        nv = cfg.n_vehicles
        self.nv = nv
        self.serving = sc.meas[0].argmax(axis=1).astype(np.int64)
        self.last_ho = np.full(nv, -300.0)        # v2 convention
        self.last_change = self.last_ho.copy()    # HO or (Phase 0b) RLF re-establishment
        self.prev_source = np.full(nv, -1, np.int64)
        self.ho_times: list[list[float]] = [[] for _ in range(nv)]
        self.blacklist_until = np.full(nv, -1.0)
        self.rlf_timer = np.zeros(nv, np.int64)
        self.hist = np.repeat(sc.meas[0][np.arange(nv), self.serving][:, None], 5, axis=1)
        self.c = dict(n_ho=0, n_hof=0, n_pp1=0, n_pp10=0, n_rlf=0, rlf_too_late=0,
                      rlf_too_early=0, rlf_wrong_cell=0, rlf_coverage=0, outage_steps=0,
                      ue_steps=0, sum_rsrp=0.0, sum_reward=0.0, n_reward=0)
        self.sinr_log: list[np.ndarray] = []
        self.last_state = self.last_action = self.last_reward = None
        self.events: list[tuple] = []             # (kind, cell a, cell b, cell c) for MRO

    def _counts(self, t_s: float, window: float) -> np.ndarray:
        return np.asarray([sum(t_s - x <= window for x in v) for v in self.ho_times])

    def observe(self, t: int) -> None:
        sc, cfg, idx = self.sc, self.cfg, np.arange(self.nv)
        t_s = (t + 1) * cfg.dt_s
        sinr = sc.sinr_true[t, idx, self.serving]
        self.sinr_log.append(sinr.copy())
        self.c["outage_steps"] += int((sinr < cfg.q_out_db).sum())
        if cfg.failure_aware_reward and self.last_reward is not None:
            pen = cfg.outage_penalty * (sinr < cfg.q_out_db)
            self.last_reward = self.last_reward + pen.astype(np.float32)
            self.c["sum_reward"] += float(pen.sum())
        self.c["ue_steps"] += self.nv
        self.c["sum_rsrp"] += float(sc.truth[t, idx, self.serving].sum())
        self.rlf_timer = np.where(sinr < cfg.q_out_db, self.rlf_timer + 1,
                                  np.where(sinr > cfg.q_in_db, 0, self.rlf_timer))
        for v in np.flatnonzero(self.rlf_timer >= cfg.t310_steps):
            new = int(sc.truth[t, v].argmax())
            recent = (t_s - self.last_ho[v]) <= cfg.t_store_s
            src, srv = int(self.prev_source[v]), int(self.serving[v])
            if recent and new == self.prev_source[v]:
                self.c["rlf_too_early"] += 1
                self.events.append(("too_early", src, srv, new))
            elif recent and new != self.serving[v]:
                self.c["rlf_wrong_cell"] += 1
                self.events.append(("wrong_cell", src, srv, new))
            elif not recent and new != self.serving[v]:
                self.c["rlf_too_late"] += 1
                self.events.append(("too_late", srv, new, -1))
            else:
                self.c["rlf_coverage"] += 1
            self.c["n_rlf"] += 1
            self.serving[v] = new
            self.rlf_timer[v] = 0
            if cfg.reestablish_is_change:
                self.last_change[v] = t_s
            if cfg.failure_aware_reward and self.last_reward is not None:
                self.last_reward[v] += cfg.rlf_penalty
                self.c["sum_reward"] += cfg.rlf_penalty
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = sc.meas[t, idx, self.serving]

    def source(self, t: int, k: int) -> np.ndarray:
        return self.sc.meas[t + k]

    def states(self, t: int, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sc, cfg, idx = self.sc, self.cfg, np.arange(self.nv)
        t_s = (t + 1) * cfg.dt_s
        norm = lambda r: np.clip((r + cfg.rsrp_norm_offset) / cfg.rsrp_norm_scale, 0, 1)
        meas = sc.meas[t]
        cur = meas[idx, self.serving]
        nb = x.copy()
        nb[idx, self.serving] = -np.inf
        nb[meas < cfg.r_min_dbm] = -np.inf
        order = np.argsort(nb, axis=1)[:, ::-1]
        top = np.take_along_axis(nb, order[:, :3], axis=1)
        top = np.where(np.isfinite(top), top, -cfg.rsrp_norm_offset)
        best = np.where(np.isfinite(nb[idx, order[:, 0]]), order[:, 0], self.serving)
        x_srv = x[idx, self.serving]
        dist = np.hypot(sc.x[t] - sc.cells_x[self.serving], sc.y[t] - sc.cells_y[self.serving])
        time_since = t_s - self.last_change
        c60 = self._counts(t_s, cfg.recent_window_s)
        sinr_m = sc.sinr_meas[t, idx, self.serving]
        if k == 0:
            trend = np.tanh((self.hist[:, -1] - self.hist[:, -3]) / 10.0)
        else:
            trend = np.tanh((x_srv - cur) / 10.0)
        velocity = np.tanh((self.hist[:, -1] - self.hist[:, 0]) / 25.0)
        s = np.column_stack([
            norm(cur), norm(top[:, 0]), norm(top[:, 1]), norm(top[:, 2]),
            np.maximum(0, (top[:, 0] - x_srv) / 20),
            signal_quality_numeric(cur),
            np.minimum(dist / 2000, 1),
            sc.speed[t] / 100,
            np.cos(np.radians(sc.heading[t])), np.sin(np.radians(sc.heading[t])),
            np.minimum(time_since / 300, 1),
            np.minimum(c60 / 10, 1),
            np.zeros(self.nv),                                   # s13: no forecaster in Phase 0
            np.minimum(sc.loads[t][self.serving] / 30, 1),
            np.clip((sinr_m + 10) / 40, 0, 1),
            (x > -100).mean(axis=1),
            trend, velocity,
        ]).astype(np.float32)
        return np.clip(s, -3, 3), best

    def override(self, t: int) -> np.ndarray:
        """Phase 0c: the link is in danger, so the anti-ping-pong limits don't apply."""
        cfg = self.cfg
        if not cfg.reliability_override:
            return np.zeros(self.nv, bool)
        sinr_m = self.sc.sinr_meas[t, np.arange(self.nv), self.serving]
        return sinr_m < cfg.q_out_db + cfg.override_margin_db

    def forced_stay(self, t: int) -> np.ndarray:
        cfg = self.cfg
        t_s = (t + 1) * cfg.dt_s
        c300 = self._counts(t_s, 300.0)
        c60 = self._counts(t_s, 60.0)
        bad = (c300 > cfg.blacklist_count_300s) | (c60 > cfg.blacklist_count_60s)
        self.blacklist_until = np.where(bad, np.maximum(self.blacklist_until,
                                                        t_s + cfg.blacklist_duration_s),
                                        self.blacklist_until)
        return (((t_s - self.last_ho < cfg.min_handover_interval_s) |
                 (t_s < self.blacklist_until)) & ~self.override(t))

    def reward(self, t: int, action: np.ndarray, target: np.ndarray) -> np.ndarray:
        sc, cfg, idx = self.sc, self.cfg, np.arange(self.nv)
        t_s = (t + 1) * cfg.dt_s
        if cfg.reward_mode == "kpi":
            return np.zeros(self.nv, np.float32)
        return shaped_reward(cfg, action, sc.truth[t, idx, self.serving],
                             sc.truth[t, idx, target], t_s - self.last_change,
                             sc.loads[t][target], self._counts(t_s, 300.0),
                             (t_s < self.blacklist_until) & ~self.override(t))

    def execute(self, t: int, execute_mask: np.ndarray, target: np.ndarray) -> np.ndarray:
        sc, cfg = self.sc, self.cfg
        t_s = (t + 1) * cfg.dt_s
        failed = np.zeros(self.nv, bool)
        for v in np.flatnonzero(execute_mask & (target != self.serving)):
            src, dst = int(self.serving[v]), int(target[v])
            self.c["n_ho"] += 1
            if sc.sinr_true[t, v, dst] < cfg.q_out_db:
                self.c["n_hof"] += 1
                failed[v] = True
                new = int(sc.truth[t, v].argmax())
            else:
                new = dst
                stay = t_s - self.last_ho[v]
                if dst == self.prev_source[v]:
                    self.c["n_pp1"] += int(stay < cfg.pp_mts_s)
                    if stay < cfg.pp_mts_s:
                        self.events.append(("ping_pong", dst, src, -1))
                    self.c["n_pp10"] += int(stay < cfg.pp_secondary_s)
            self.prev_source[v] = src
            self.last_ho[v] = t_s
            self.last_change[v] = t_s
            self.ho_times[v].append(t_s)
            self.serving[v] = new
            self.rlf_timer[v] = 0
        return failed

    def gate(self, t: int, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Constraint-layer execution gate on the arm's own signal source."""
        cfg, idx = self.cfg, np.arange(self.nv)
        t_s = (t + 1) * cfg.dt_s
        gain = x[idx, target] - x[idx, self.serving]
        limits = ((t_s - self.last_ho >= cfg.min_handover_interval_s) &
                  (t_s >= self.blacklist_until)) | self.override(t)
        ok = limits & (gain >= cfg.min_rsrp_improvement_db)
        if cfg.admission_check:
            ok &= self.sc.sinr_meas[t, idx, target] >= cfg.q_in_db
        return ok

    def step_agent(self, agent: DQNAgent, t: int, k: int, training: bool) -> np.ndarray:
        self.observe(t)
        x = self.source(t, k)
        s, target = self.states(t, x, k)
        if training and self.last_state is not None:
            agent.remember_batch(self.last_state, self.last_action, self.last_reward, s,
                                 np.zeros(self.nv, bool))
        action = agent.select(s, self.forced_stay(t), training)
        r = self.reward(t, action, target)
        mask = (action == 1) & self.gate(t, x, target)
        if self.cfg.reward_mode == "kpi":
            t_s = (t + 1) * self.cfg.dt_s
            done = mask & (target != self.serving)
            pp = done & (target == self.prev_source) & (t_s - self.last_ho < self.cfg.pp_mts_s)
            r = (-self.cfg.ho_cost * done - self.cfg.pp_penalty * pp).astype(np.float32)
        failed = self.execute(t, mask, target)
        if self.cfg.failure_aware_reward:
            r = r + (self.cfg.hof_penalty * failed).astype(np.float32)
        self.c["sum_reward"] += float(r.sum())
        self.c["n_reward"] += len(r)
        self.last_state, self.last_action, self.last_reward = s, action, r
        return r

    def step_rules(self, t: int) -> None:
        self.observe(t)
        x = self.source(t, 0)
        _, target = self.states(t, x, 0)
        forced = self.forced_stay(t)
        self.execute(t, (~forced) & self.gate(t, x, target), target)

    def metrics(self, arm: str, training_seed, scenario_seed: int) -> dict:
        c, cfg = self.c, self.cfg
        ue_hours = c["ue_steps"] * cfg.dt_s / 3600
        sinr = np.concatenate(self.sinr_log)
        return {
            "arm": arm, "training_seed": training_seed, "scenario_seed": scenario_seed,
            **{k: v for k, v in c.items() if k not in ("sum_rsrp", "sum_reward", "n_reward")},
            "hsr": 1 - c["n_hof"] / c["n_ho"] if c["n_ho"] else float("nan"),
            "hof_rate": c["n_hof"] / c["n_ho"] if c["n_ho"] else 0.0,
            "ppr1": c["n_pp1"] / c["n_ho"] if c["n_ho"] else 0.0,
            "ppr10": c["n_pp10"] / c["n_ho"] if c["n_ho"] else 0.0,
            "rlf_per_ueh": c["n_rlf"] / ue_hours,
            "ho_per_ueh": c["n_ho"] / ue_hours,
            "outage_frac": c["outage_steps"] / c["ue_steps"],
            "mean_sinr_db": float(sinr.mean()),
            "p5_sinr_db": float(np.percentile(sinr, 5)),
            "mean_rsrp_dbm": c["sum_rsrp"] / c["ue_steps"],
            "mean_reward": c["sum_reward"] / c["n_reward"] if c["n_reward"] else float("nan"),
        }


class A3Controller:
    """TS 38.331 A3 on L3-filtered RSRP: Mn - Hys > Mp + Off held for TTT."""

    def __init__(self, cfg: TwinConfig, margin_db: float, ttt_s: float):
        self.cfg, self.margin, self.ttt = cfg, margin_db, ttt_s
        self.trig_cell = np.full(cfg.n_vehicles, -1, np.int64)
        self.trig_start = np.zeros(cfg.n_vehicles)

    def step(self, rt: Runtime, t: int) -> None:
        rt.observe(t)
        sc, cfg = rt.sc, self.cfg
        t_s = (t + 1) * cfg.dt_s
        m = sc.meas[t].copy()
        idx = np.arange(rt.nv)
        srv = m[idx, rt.serving]
        m[idx, rt.serving] = -np.inf
        best = m.argmax(axis=1)
        ok = m[idx, best] > srv + self.margin
        fire = np.zeros(rt.nv, bool)
        for v in range(rt.nv):
            if not ok[v]:
                self.trig_cell[v] = -1
            elif self.trig_cell[v] != best[v]:
                self.trig_cell[v], self.trig_start[v] = best[v], t_s
            elif t_s - self.trig_start[v] >= self.ttt - 1e-9:
                fire[v] = True
                self.trig_cell[v] = -1
        rt.execute(t, fire, best)
