"""Shared, auditable Chapter-5 experiment components.

This module contains no plotting constants and no paper-result constants.  Every
reported value is produced by a run or by predictor inference on the held-out
trace split.  The implementation preserves the recovered 18-feature state,
network, Double-DQN/PER update, reward, constraints, mobility, and propagation
contracts while using simulated time (rather than wall-clock time) throughout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
import torch.nn as nn


SOURCE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_ROOT = SOURCE_ROOT / "retraining_pipeline"
sys.path.insert(0, str(PREDICTOR_ROOT))

from architectures import build_model  # noqa: E402


ARM_PROACTIVE = "Proactive (LSTM + DQN)"
ARM_REACTIVE = "Reactive DQN"
ARM_A3 = "Static A3"

L3_FILTER_K = 4

STATE_FEATURES = [
    "Serving signal level",
    "Best candidate signal level",
    "Second-best candidate signal level",
    "Third-best candidate signal level",
    "Best candidate gain",
    "Serving signal quality",
    "Distance to serving cell",
    "Vehicle speed",
    "Heading cosine",
    "Heading sine",
    "Time since handover",
    "Recent handover frequency",
    "Time-of-day indicator",
    "Serving-cell load",
    "Interference context",
    "Strong-cell fraction",
    "Signal trend",
    "Signal velocity",
]


@dataclass(frozen=True)
class PinnedConfig:
    grid_width_m: int = 2000
    grid_height_m: int = 1500
    n_macro: int = 7
    n_micro: int = 4
    n_pico: int = 1
    n_vehicles: int = 20
    dt_s: float = 0.2
    steps_per_episode: int = 1000
    training_episodes: int = 100
    evaluation_scenarios: int = 10
    training_seed: int = 42
    evaluation_seed_start: int = 1000
    layout_seed: int = 42
    speed_spawn_min_kmh: float = 30.0
    speed_spawn_max_kmh: float = 80.0
    speed_clip_min_kmh: float = 20.0
    speed_clip_max_kmh: float = 100.0
    state_size: int = 18
    action_size: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 5e-5
    gamma: float = 0.995
    epsilon_start: float = 0.95
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    batch_size: int = 128
    replay_capacity: int = 100000
    target_update_steps: int = 1000
    warmup_updates: int = 10000
    priority_alpha: float = 0.7
    importance_exponent: float = -0.4
    gradient_clip_norm: float = 1.0
    min_handover_interval_s: float = 30.0
    min_rsrp_improvement_db: float = 3.0
    blacklist_duration_s: float = 120.0
    blacklist_count_300s: int = 4
    blacklist_count_60s: int = 2
    a3_offset_db: float = 0.0
    a3_hysteresis_db: float = 3.0
    a3_ttt_s: float = 0.16
    rlf_threshold_dbm: float = -118.0
    rlf_consecutive_measurements: int = 3
    rlf_recovery_db: float = 0.0
    ping_pong_window_s: float = 60.0
    predictor_history: int = 15
    predictor_input_features: int = 12
    predictor_output_steps: int = 1
    predictor_disagreement_limit_db: float = 15.0
    predictor_min_confidence: float = 0.3

    @property
    def n_cells(self) -> int:
        return self.n_macro + self.n_micro + self.n_pico

    @property
    def evaluation_seeds(self) -> list[int]:
        return list(range(self.evaluation_seed_start,
                          self.evaluation_seed_start + self.evaluation_scenarios))


@dataclass(frozen=True)
class Cell:
    cell_id: int
    cell_type: str
    x_m: float
    y_m: float
    tx_power_dbm: float
    coverage_radius_m: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cells(cfg: PinnedConfig) -> list[Cell]:
    """Create the pinned 7+4+1 layout.

    The centre macro plus six macros on a 600 m ring gives exactly 600 m
    centre/ring and adjacent-ring spacing in the 2,000 x 1,500 m grid.
    """
    cells: list[Cell] = []
    cx, cy, radius = cfg.grid_width_m / 2, cfg.grid_height_m / 2, 600.0
    cells.append(Cell(0, "macro", cx, cy, 46.0, 1500.0))
    for i in range(6):
        angle = 2 * math.pi * i / 6
        cells.append(Cell(i + 1, "macro", cx + radius * math.cos(angle),
                          cy + radius * math.sin(angle), 46.0, 1500.0))

    # Four small cells are placed near alternating ring macros.  The offsets
    # are deterministic consequences of the pinned layout seed.
    rng = np.random.default_rng(cfg.layout_seed)
    for i, macro_id in enumerate((1, 2, 4, 5)):
        ref = cells[macro_id]
        angle = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(180, 300)
        x = float(np.clip(ref.x_m + distance * math.cos(angle), 50, cfg.grid_width_m - 50))
        y = float(np.clip(ref.y_m + distance * math.sin(angle), 50, cfg.grid_height_m - 50))
        cells.append(Cell(7 + i, "micro", x, y, 38.0, 500.0))

    x = float(rng.uniform(100, cfg.grid_width_m - 100))
    y = float(rng.uniform(100, cfg.grid_height_m - 100))
    cells.append(Cell(11, "pico", x, y, 30.0, 200.0))
    assert len(cells) == cfg.n_cells
    return cells


class DQNNetwork(nn.Module):
    """Recovered 18 -> 512 -> 256 -> 128 -> 64 -> 2 decision network."""

    def __init__(self, state_size: int = 18, action_size: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 512), nn.ReLU(), nn.LayerNorm(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SumTreeReplay:
    """Fixed-capacity proportional replay with exact priority probabilities."""

    def __init__(self, capacity: int, state_size: int, alpha: float, seed: int):
        self.capacity = capacity
        self.state_size = state_size
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, state_size), np.float32)
        self.next_states = np.empty((capacity, state_size), np.float32)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.dones = np.empty(capacity, np.bool_)
        tree_n = 1
        while tree_n < capacity:
            tree_n *= 2
        self.tree_n = tree_n
        self.tree = np.zeros(2 * tree_n, np.float64)
        self.raw_priorities = np.zeros(capacity, np.float64)
        self.pos = 0
        self.size = 0

    def _set_leaf(self, idx: int, raw_priority: float) -> None:
        value = float(raw_priority) ** self.alpha
        self.raw_priorities[idx] = raw_priority
        node = self.tree_n + idx
        delta = value - self.tree[node]
        while node:
            self.tree[node] += delta
            node //= 2

    def add_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                  next_states: np.ndarray, dones: np.ndarray) -> None:
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            idx = self.pos
            self.states[idx] = s
            self.actions[idx] = a
            self.rewards[idx] = np.clip(r, -50, 10)
            self.next_states[idx] = ns
            self.dones[idx] = d
            self._set_leaf(idx, abs(float(r)) + 0.1)
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, importance_exponent: float):
        total = self.tree[1]
        if total <= 0:
            raise RuntimeError("Replay priorities sum to zero")
        draws = self.rng.random(batch_size) * total
        idx = np.empty(batch_size, np.int64)
        for j, draw in enumerate(draws):
            node = 1
            while node < self.tree_n:
                left = 2 * node
                if draw <= self.tree[left]:
                    node = left
                else:
                    draw -= self.tree[left]
                    node = left + 1
            idx[j] = min(node - self.tree_n, self.size - 1)
        probs = self.tree[self.tree_n + idx] / total
        weights = (self.size * probs) ** importance_exponent
        weights /= weights.max()
        return idx, weights.astype(np.float32)

    def update(self, idx: np.ndarray, priorities: np.ndarray) -> None:
        for i, p in zip(idx, priorities):
            self._set_leaf(int(i), float(p))


class DQNAgent:
    def __init__(self, cfg: PinnedConfig, seed: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        set_global_seed(seed)
        self.q = DQNNetwork(cfg.state_size, cfg.action_size).to(device)
        # The recovered implementation creates an independently initialized
        # target and synchronizes it on the first scheduled hard update.
        self.target = DQNNetwork(cfg.state_size, cfg.action_size).to(device)
        self.optimizer = torch.optim.AdamW(self.q.parameters(), lr=cfg.learning_rate,
                                           weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=cfg.learning_rate / 50)
        self.replay = SumTreeReplay(cfg.replay_capacity, cfg.state_size,
                                    cfg.priority_alpha, seed + 171)
        self.rng = np.random.default_rng(seed + 313)
        self.epsilon = cfg.epsilon_start
        self.updates = 0
        self.losses: list[float] = []
        self.recent_rewards: deque[float] = deque(maxlen=5000)

    def select(self, states: np.ndarray, forced_stay: np.ndarray, training: bool) -> np.ndarray:
        n = len(states)
        actions = np.zeros(n, np.int64)
        eligible = np.flatnonzero(~forced_stay)
        if not len(eligible):
            return actions
        if training and self.updates < self.cfg.warmup_updates:
            actions[eligible] = self.rng.integers(0, 2, size=len(eligible))
            return actions
        explore = np.zeros(len(eligible), dtype=bool)
        if training:
            explore = self.rng.random(len(eligible)) < self.epsilon
            actions[eligible[explore]] = self.rng.integers(0, 2, size=int(explore.sum()))
        exploit_idx = eligible[~explore]
        if len(exploit_idx):
            self.q.eval()
            with torch.no_grad():
                x = torch.from_numpy(states[exploit_idx]).to(self.device)
                actions[exploit_idx] = self.q(x).argmax(1).cpu().numpy()
            if training:
                self.q.train()
        return actions

    def remember_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                       next_states: np.ndarray, dones: np.ndarray) -> None:
        self.replay.add_batch(states, actions, rewards, next_states, dones)
        self.recent_rewards.extend(float(r) for r in rewards)

    def train_update(self) -> float | None:
        if self.replay.size < self.cfg.batch_size:
            return None
        idx, weights = self.replay.sample(self.cfg.batch_size, self.cfg.importance_exponent)
        states = torch.from_numpy(self.replay.states[idx]).to(self.device)
        actions = torch.from_numpy(self.replay.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.replay.rewards[idx]).to(self.device)
        next_states = torch.from_numpy(self.replay.next_states[idx]).to(self.device)
        dones = torch.from_numpy(self.replay.dones[idx]).to(self.device)
        w = torch.from_numpy(weights).to(self.device)

        self.q.train()
        current = self.q(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.q(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
            target = rewards + self.cfg.gamma * next_q * (~dones)
        td = current - target
        loss = (w * td.square()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.gradient_clip_norm)
        self.optimizer.step()
        self.replay.update(idx, np.abs(td.detach().cpu().numpy()) + 0.1)

        self.updates += 1
        if self.updates > self.cfg.warmup_updates:
            self.scheduler.step()
            if self.epsilon > self.cfg.epsilon_min:
                self.epsilon = max(self.cfg.epsilon_min,
                                   self.epsilon * self.cfg.epsilon_decay)
        if self.updates % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.q.state_dict())
        value = float(loss.item())
        self.losses.append(value)
        return value

    def converged(self) -> bool:
        if len(self.recent_rewards) < 3000:
            return False
        values = np.asarray(list(self.recent_rewards)[-3000:], dtype=float)
        mean = values.mean()
        cv = values.std() / abs(mean) if abs(mean) > 0.01 else np.inf
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        return bool(cv < 0.15 and slope >= -0.0005 and mean > -2.0)

    def save(self, path: Path) -> None:
        torch.save({"q_state_dict": self.q.state_dict(),
                    "target_state_dict": self.target.state_dict(),
                    "epsilon": self.epsilon, "updates": self.updates}, path)

    def load(self, path: Path) -> None:
        item = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(item["q_state_dict"])
        self.target.load_state_dict(item["target_state_dict"])
        self.epsilon = float(item["epsilon"])
        self.updates = int(item["updates"])
        self.q.eval()


class ScenarioEpisode:
    """Policy-neutral mobility and radio channel for one measured episode.

    One hybrid predictor instance produces the radio matrix consumed as both the
    environment channel and, before Task-E noise, the proactive prediction
    channel.  Predictor feature history is kept per vehicle/cell and does not
    consume policy state, making the truth channel invariant across arms.
    """

    def __init__(self, cfg: PinnedConfig, cells: list[Cell], predictor: nn.Module,
                 scaler_artifact: dict, seed: int, device: torch.device):
        self.cfg = cfg
        self.cells = cells
        self.predictor = predictor
        self.scaler = scaler_artifact["feature_scaler"]
        self.non_rsrp_cols = scaler_artifact["non_rsrp_cols"]
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.nv, self.nc = cfg.n_vehicles, cfg.n_cells
        self.x = np.empty(self.nv)
        self.y = np.empty(self.nv)
        self.speed = self.rng.uniform(cfg.speed_spawn_min_kmh,
                                      cfg.speed_spawn_max_kmh, self.nv)
        self.heading = self.rng.uniform(0, 360, self.nv)
        self.loads = self.rng.integers(0, 11, self.nc).astype(float)
        self._spawn_positions()
        self.overlap_penalty = self._overlap_penalties()
        self.shadow_std = np.asarray([12.0 if c.cell_type == "macro" else
                                      8.0 if c.cell_type == "micro" else 6.0
                                      for c in self.cells])[None, :]
        # F1: decorrelation distances from 3GPP TR 38.901 Table 7.5-6 (UMa/UMi NLOS).
        self.shadow_dcorr = np.asarray([50.0 if c.cell_type == "macro" else 13.0
                                        for c in self.cells])[None, :]
        self.shadow = self.rng.normal(0, 1, (self.nv, self.nc)) * self.shadow_std
        initial_math, distance = self._math_rsrp()
        # F2: TS 38.331 5.5.3.2 initialises the L3 filter with the first measurement.
        self.l3_coefficient = 1.0 / 2 ** (L3_FILTER_K / 4)
        self.measured = initial_math.copy()
        self.truth_prev = initial_math
        self.truth_hist = np.repeat(initial_math[:, :, None], 5, axis=2)
        first = self._normalized_features(distance, 0.0)
        self.sequence = np.repeat(first[:, :, None, :], cfg.predictor_history, axis=2)

    def _spawn_positions(self) -> None:
        for i in range(self.nv):
            if self.rng.random() < 0.8:
                cell = self.cells[int(self.rng.integers(0, self.nc))]
                angle = self.rng.uniform(0, 2 * math.pi)
                if cell.cell_type == "macro":
                    distance = self.rng.uniform(100, 1000)
                elif cell.cell_type == "micro":
                    distance = self.rng.uniform(50, 500)
                else:
                    distance = self.rng.uniform(20, 250)
                self.x[i] = np.clip(cell.x_m + distance * math.cos(angle),
                                    0, self.cfg.grid_width_m)
                self.y[i] = np.clip(cell.y_m + distance * math.sin(angle),
                                    0, self.cfg.grid_height_m)
            else:
                self.x[i] = self.rng.uniform(0, self.cfg.grid_width_m)
                self.y[i] = self.rng.uniform(0, self.cfg.grid_height_m)

    def _overlap_penalties(self) -> np.ndarray:
        out = np.zeros(self.nc)
        for i, cell in enumerate(self.cells):
            for j, other in enumerate(self.cells):
                if i == j:
                    continue
                distance = math.hypot(cell.x_m - other.x_m, cell.y_m - other.y_m)
                if cell.cell_type == other.cell_type == "macro":
                    threshold = 1000.0
                elif "micro" in (cell.cell_type, other.cell_type):
                    threshold = 600.0
                else:
                    threshold = 400.0
                if distance < threshold:
                    out[i] += 0.8 * (1 - distance / threshold)
        return out

    def _distances(self) -> np.ndarray:
        cx = np.asarray([c.x_m for c in self.cells])[None, :]
        cy = np.asarray([c.y_m for c in self.cells])[None, :]
        return np.hypot(self.x[:, None] - cx, self.y[:, None] - cy)

    def _math_rsrp(self) -> tuple[np.ndarray, np.ndarray]:
        distance = self._distances()
        intercept = np.asarray([140.1 if c.cell_type == "macro" else
                                145.7 if c.cell_type == "micro" else 150.4
                                for c in self.cells])[None, :]
        slope = np.asarray([42.6 if c.cell_type == "macro" else
                            40.2 if c.cell_type == "micro" else 38.5
                            for c in self.cells])[None, :]
        fading_std = np.asarray([4.0 if c.cell_type == "macro" else
                                 3.5 if c.cell_type == "micro" else 4.0
                                 for c in self.cells])[None, :]
        power = np.asarray([c.tx_power_dbm for c in self.cells])[None, :]
        env = np.where(self.speed[:, None] > 60, 1.1,
                       np.where(self.speed[:, None] < 5, 0.95, 1.0))
        path_loss = (intercept + slope * np.log10(np.maximum(distance, 10) / 1000.0)) * env
        fading = self.rng.normal(0, fading_std, (self.nv, self.nc))
        rsrp = (power - path_loss + self.shadow + fading - self.overlap_penalty[None, :]
                - 0.5 * self.loads[None, :])
        return np.clip(rsrp, -120, -60), distance

    def _normalized_features(self, distance: np.ndarray, t_s: float) -> np.ndarray:
        rows = self.nv * self.nc
        # Map the simulation grid monotonically onto the trace scaler's observed
        # coordinate support.  This preserves the recovered [0,1] spatial inputs
        # without claiming the simulated grid has geographic coordinates.
        lon_i = self.non_rsrp_cols.index("Longitude")
        lat_i = self.non_rsrp_cols.index("Latitude")
        lon_raw = self.scaler.data_min_[lon_i] + (self.x / self.cfg.grid_width_m) * (
            self.scaler.data_max_[lon_i] - self.scaler.data_min_[lon_i])
        lat_raw = self.scaler.data_min_[lat_i] + (self.y / self.cfg.grid_height_m) * (
            self.scaler.data_max_[lat_i] - self.scaler.data_min_[lat_i])
        hour = 8.0 + t_s / 3600.0
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        trend = (self.truth_hist[:, :, -1] - self.truth_hist[:, :, -2]) / 60.0
        raw = {
            "Longitude": np.repeat(lon_raw[:, None], self.nc, axis=1),
            "Latitude": np.repeat(lat_raw[:, None], self.nc, axis=1),
            "Speed": np.repeat(self.speed[:, None], self.nc, axis=1),
            "Distance_serving": distance,
            "Heading": np.repeat(self.heading[:, None], self.nc, axis=1),
            "Hour_sin": np.full((self.nv, self.nc), hour_sin),
            "Hour_cos": np.full((self.nv, self.nc), hour_cos),
            "Handover_recent": np.zeros((self.nv, self.nc)),
            "RSRP_trend": trend,
            "Speed_normalized": np.repeat((self.speed / 120.0)[:, None], self.nc, axis=1),
            "Distance_normalized": np.clip(distance / 2000.0, 0, 1),
        }
        raw_matrix = np.column_stack([raw[name].reshape(rows) for name in self.non_rsrp_cols])
        scaled = self.scaler.transform(raw_matrix)
        # Feature order recovered by the fitted artifact.
        result = np.empty((rows, 12), np.float32)
        feature_order = ["Longitude", "Latitude", "Speed", "RSRP", "Distance_serving",
                         "Heading", "Hour_sin", "Hour_cos", "Handover_recent",
                         "RSRP_trend", "Speed_normalized", "Distance_normalized"]
        for j, name in enumerate(feature_order):
            if name == "RSRP":
                result[:, j] = ((self.truth_prev.reshape(rows) + 120.0) / 60.0).clip(0, 1)
            else:
                result[:, j] = scaled[:, self.non_rsrp_cols.index(name)]
        return result.reshape(self.nv, self.nc, 12)

    def step(self, step: int) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        t_s = (step + 1) * self.cfg.dt_s
        self.heading = (self.heading + self.rng.uniform(-10, 10, self.nv)) % 360
        self.speed = np.clip(self.speed + self.rng.uniform(-5, 5, self.nv),
                             self.cfg.speed_clip_min_kmh, self.cfg.speed_clip_max_kmh)
        distance_moved = self.speed * self.cfg.dt_s / 3.6
        nx = self.x + distance_moved * np.cos(np.radians(self.heading))
        ny = self.y + distance_moved * np.sin(np.radians(self.heading))
        hit_x = (nx < 0) | (nx > self.cfg.grid_width_m)
        hit_y = (ny < 0) | (ny > self.cfg.grid_height_m)
        self.heading[hit_x] = (180 - self.heading[hit_x]) % 360
        self.heading[hit_y] = (-self.heading[hit_y]) % 360
        self.x = np.clip(nx, 0, self.cfg.grid_width_m)
        self.y = np.clip(ny, 0, self.cfg.grid_height_m)
        rho = np.exp(-distance_moved[:, None] / self.shadow_dcorr)
        innovation = self.rng.normal(0, 1, (self.nv, self.nc)) * self.shadow_std
        self.shadow = rho * self.shadow + np.sqrt(1 - rho ** 2) * innovation
        if step % 50 == 0:
            change = self.rng.integers(-2, 4, self.nc)
            active = self.rng.random(self.nc) < 0.1
            self.loads = np.clip(self.loads + change * active, 0, 30)

        math_rsrp, distance = self._math_rsrp()
        features = self._normalized_features(distance, t_s)
        self.sequence = np.roll(self.sequence, -1, axis=2)
        self.sequence[:, :, -1, :] = features
        batch = torch.from_numpy(self.sequence.reshape(self.nv * self.nc,
                                                       self.cfg.predictor_history, 12)).to(self.device)
        self.predictor.eval()
        with torch.no_grad():
            pred_norm = self.predictor(batch).reshape(self.nv, self.nc).cpu().numpy()
        lstm_rsrp = np.clip(pred_norm * 60.0 - 120.0, -120, -60)
        error = np.abs(lstm_rsrp - math_rsrp)
        confidence = 1.0 / (1.0 + error / 8.0)
        use_hybrid = ((error < self.cfg.predictor_disagreement_limit_db) &
                      (confidence > self.cfg.predictor_min_confidence))
        truth = np.where(use_hybrid,
                         confidence * lstm_rsrp + (1 - confidence) * math_rsrp,
                         math_rsrp)
        truth = np.clip(truth, -120, -60)
        self.truth_prev = truth
        self.truth_hist = np.roll(self.truth_hist, -1, axis=2)
        self.truth_hist[:, :, -1] = truth
        a = self.l3_coefficient
        self.measured = (1 - a) * self.measured + a * truth
        meta = {
            "measured": self.measured.copy(),
            "t_s": t_s,
            "x": self.x.copy(),
            "y": self.y.copy(),
            "speed": self.speed.copy(),
            "heading": self.heading.copy(),
            "distance": distance,
            "loads": self.loads.copy(),
            "interference_feature": self.rng.uniform(0.0, 0.3, self.nv),
            "hybrid_fraction": float(use_hybrid.mean()),
        }
        return truth, meta


def signal_quality_numeric(rsrp: np.ndarray) -> np.ndarray:
    return np.select([rsrp > -70, rsrp > -85, rsrp > -100, rsrp > -110],
                     [1.0, 0.8, 0.5, 0.2], default=0.0)


def signal_reward(rsrp: np.ndarray) -> np.ndarray:
    return np.select([rsrp > -70, rsrp > -85, rsrp > -100, rsrp > -110],
                     [1.0, 0.5, 0.0, -0.5], default=-1.0)


def shaped_reward(cfg: PinnedConfig, action: np.ndarray, current: np.ndarray,
                  target: np.ndarray, time_since: np.ndarray, target_load: np.ndarray,
                  recent_count: np.ndarray, blacklisted: np.ndarray) -> np.ndarray:
    current = np.clip(current, -120, -60)
    target = np.clip(target, -120, -60)
    base = signal_reward(current)
    result = np.empty(len(current), float)
    ho = action == 1
    stay = ~ho
    if ho.any():
        gain = target[ho] - current[ho]
        low_gain = gain < cfg.min_rsrp_improvement_db
        gain_penalty = np.where(low_gain, -5.0, 0.0)
        gain_value = np.where(low_gain, np.maximum(gain / 2.0, -6.0),
                              np.minimum(gain / 3.0, 3.0))
        ts = time_since[ho]
        pp = np.select([ts < 10, ts < cfg.min_handover_interval_s, ts < 60, ts < 120],
                       [-20.0, -10.0, -4.0, -1.5], default=0.0)
        load_penalty = -1.2 * np.minimum(target_load[ho] / 15.0, 1.0)
        critical = np.where(current[ho] < -115, 0.5, 0.0)
        rc = recent_count[ho]
        frequency = np.select([rc >= 5, rc >= 3, rc >= 2], [-8.0, -4.0, -2.0], 0.0)
        blacklist_penalty = np.where(blacklisted[ho], -15.0, 0.0)
        pattern = np.where((ts < cfg.min_handover_interval_s) & (rc >= 3), -25.0, 0.0)
        result[ho] = (base[ho] + gain_value + critical - 2.0 + pp + load_penalty +
                      gain_penalty + frequency + blacklist_penalty + pattern)
    if stay.any():
        c = current[stay]
        stability = np.select([c > -85, c > -95, c > -105], [2.0, 1.5, 1.0], 0.0)
        degraded = np.select([c < -115, c < -110, c < -105], [-3.0, -2.0, -0.8], 0.0)
        gain = target[stay] - c
        avoid = np.select([gain < cfg.min_rsrp_improvement_db, gain < 5], [2.0, 1.0], 0.0)
        ts = time_since[stay]
        long_stable = np.select([ts > 180, ts > 120, ts > 60], [1.5, 1.0, 0.5], 0.0)
        result[stay] = base[stay] + stability + degraded + avoid + long_stable
    return np.clip(result, -30, 5).astype(np.float32)


class PolicyRuntime:
    """Per-arm connection, constraint, reward, and counter state."""

    def __init__(self, cfg: PinnedConfig, initial_truth: np.ndarray):
        self.cfg = cfg
        self.nv = cfg.n_vehicles
        cells = make_cells(cfg)
        self.cells_x = np.asarray([c.x_m for c in cells])
        self.cells_y = np.asarray([c.y_m for c in cells])
        self.connected = initial_truth.argmax(axis=1).astype(np.int64)
        self.last_ho = np.full(self.nv, -300.0)
        self.ho_times: list[list[float]] = [[] for _ in range(self.nv)]
        self.ho_records: list[list[tuple[int, int, float]]] = [[] for _ in range(self.nv)]
        self.blacklist_until = np.full(self.nv, -1.0)
        self.rsrp_hist = np.repeat(initial_truth[np.arange(self.nv), self.connected, None], 5, axis=1)
        self.rlf_streak = np.zeros(self.nv, np.int64)
        self.rlf_active = np.zeros(self.nv, bool)
        self.n_ho = self.n_succ = self.n_pp = self.n_conn = self.n_rlf = 0
        self.sum_rsrp = 0.0
        self.sum_reward = 0.0
        self.reward_count = 0
        self.last_state: np.ndarray | None = None
        self.last_action: np.ndarray | None = None
        self.last_reward: np.ndarray | None = None

    def _recent_counts(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        c300 = np.asarray([sum(t_s - t <= 300 for t in values) for values in self.ho_times])
        c60 = np.asarray([sum(t_s - t <= 60 for t in values) for values in self.ho_times])
        return c300, c60

    def observe(self, truth: np.ndarray, measured: np.ndarray) -> np.ndarray:
        current = truth[np.arange(self.nv), self.connected]
        self.rsrp_hist = np.roll(self.rsrp_hist, -1, axis=1)
        self.rsrp_hist[:, -1] = measured[np.arange(self.nv), self.connected]
        self.n_conn += self.nv
        self.sum_rsrp += float(current.sum())
        below = current < self.cfg.rlf_threshold_dbm
        self.rlf_streak = np.where(below, self.rlf_streak + 1, 0)
        new_event = ((self.rlf_streak >= self.cfg.rlf_consecutive_measurements) &
                     (~self.rlf_active))
        self.n_rlf += int(new_event.sum())
        self.rlf_active |= new_event
        self.rlf_active &= below
        return current

    def states(self, truth: np.ndarray, consumed: np.ndarray, meta: dict,
               reactive: bool) -> tuple[np.ndarray, np.ndarray]:
        measured = meta["measured"]
        current = measured[np.arange(self.nv), self.connected]
        source = measured if reactive else consumed
        order = np.argsort(source, axis=1)[:, ::-1]
        top = np.take_along_axis(source, order[:, :3], axis=1)
        best = order[:, 0]
        best_value = top[:, 0]
        serving_distance = np.hypot(np.asarray(meta["x"]) - self.cells_x[self.connected],
                                    np.asarray(meta["y"]) - self.cells_y[self.connected])
        t_s = float(meta["t_s"])
        c300, _ = self._recent_counts(t_s)
        time_since = t_s - self.last_ho
        loads = np.asarray(meta["loads"])[self.connected]
        trend = np.tanh((self.rsrp_hist[:, -1] - self.rsrp_hist[:, -3]) / 10.0)
        velocity = np.tanh(((self.rsrp_hist[:, -1] - self.rsrp_hist[:, 0]) / 5.0) / 5.0)
        states = np.column_stack([
            (current + 120) / 60,
            (top[:, 0] + 120) / 60,
            (top[:, 1] + 120) / 60,
            (top[:, 2] + 120) / 60,
            np.maximum(0, (best_value - current) / 20),
            signal_quality_numeric(current),
            np.minimum(serving_distance / 2000, 1),
            np.asarray(meta["speed"]) / 100,
            np.cos(np.radians(np.asarray(meta["heading"]))),
            np.sin(np.radians(np.asarray(meta["heading"]))),
            np.minimum(time_since / 300, 1),
            np.minimum(c300 / 10, 1),
            np.full(self.nv, (8 * 3600 + t_s) / 86400),
            np.minimum(loads / 30, 1),
            np.asarray(meta["interference_feature"]),
            (source > -100).mean(axis=1),
            trend,
            velocity,
        ]).astype(np.float32)
        return np.clip(states, -3, 3), best

    def forced_stay(self, t_s: float) -> np.ndarray:
        c300, c60 = self._recent_counts(t_s)
        newly_bad = (c300 > self.cfg.blacklist_count_300s) | (c60 > self.cfg.blacklist_count_60s)
        self.blacklist_until = np.where(newly_bad,
                                        np.maximum(self.blacklist_until,
                                                   t_s + self.cfg.blacklist_duration_s),
                                        self.blacklist_until)
        return ((t_s - self.last_ho < self.cfg.min_handover_interval_s) |
                (t_s < self.blacklist_until))

    def apply(self, raw_action: np.ndarray, target_cell: np.ndarray, truth: np.ndarray,
              meta: dict, enforce_learning_constraints: bool = True) -> np.ndarray:
        t_s = float(meta["t_s"])
        current = truth[np.arange(self.nv), self.connected]
        target = truth[np.arange(self.nv), target_cell]
        measured = meta["measured"]
        measured_gain = (measured[np.arange(self.nv), target_cell] -
                         measured[np.arange(self.nv), self.connected])
        c300, _ = self._recent_counts(t_s)
        time_since = t_s - self.last_ho
        target_load = np.asarray(meta["loads"])[target_cell]
        blacklisted = t_s < self.blacklist_until
        rewards = shaped_reward(self.cfg, raw_action, current, target, time_since,
                                target_load, c300, blacklisted)
        executable = (raw_action == 1) & (target_cell != self.connected)
        if enforce_learning_constraints:
            executable &= ((time_since >= self.cfg.min_handover_interval_s) &
                           (measured_gain >= self.cfg.min_rsrp_improvement_db) &
                           (~blacklisted))
        for v in np.flatnonzero(executable):
            source = int(self.connected[v])
            dest = int(target_cell[v])
            gain = float(target[v] - current[v])
            success = (gain >= 1.0 or
                       (current[v] < -115 and gain > -1.0) or
                       (target[v] > -90 and gain > -2.0))
            recent = [r for r in self.ho_records[v] if t_s - r[2] <= self.cfg.ping_pong_window_s]
            ping_pong = any(r[0] == dest and r[1] == source for r in recent)
            if len(recent) >= 3:
                targets = [r[1] for r in recent[-3:]]
                ping_pong |= targets.count(dest) >= 2
            self.n_ho += 1
            self.n_succ += int(success)
            self.n_pp += int(ping_pong)
            self.ho_times[v].append(t_s)
            self.ho_records[v].append((source, dest, t_s))
            self.last_ho[v] = t_s
            self.connected[v] = dest
        self.sum_reward += float(rewards.sum())
        self.reward_count += len(rewards)
        return rewards

    def step_agent(self, agent: DQNAgent, truth: np.ndarray, consumed: np.ndarray,
                   meta: dict, reactive: bool, training: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.observe(truth, meta["measured"])
        states, target = self.states(truth, consumed, meta, reactive)
        if training and self.last_state is not None:
            agent.remember_batch(self.last_state, self.last_action, self.last_reward,
                                 states, np.zeros(self.nv, bool))
        action = agent.select(states, self.forced_stay(float(meta["t_s"])), training)
        rewards = self.apply(action, target, truth, meta)
        self.last_state = states.copy()
        self.last_action = action.copy()
        self.last_reward = rewards.copy()
        return states, action, rewards

    def metrics(self, arm: str, training_seed: int | str,
                scenario_seed: int) -> dict[str, float | int | str]:
        return {
            "arm": arm,
            "training_seed": training_seed,
            "scenario_seed": scenario_seed,
            "n_ho": self.n_ho,
            "n_succ": self.n_succ,
            "n_pp": self.n_pp,
            "n_conn": self.n_conn,
            "n_rlf": self.n_rlf,
            "hsr": self.n_succ / self.n_ho if self.n_ho else 0.0,
            "ppr": self.n_pp / self.n_ho if self.n_ho else 0.0,
            "rlf": self.n_rlf / self.n_conn if self.n_conn else 0.0,
            "total_ho": self.n_ho,
            "mean_rsrp_dbm": self.sum_rsrp / self.n_conn if self.n_conn else np.nan,
            "mean_reward": self.sum_reward / self.reward_count if self.reward_count else np.nan,
        }


class StaticA3Controller:
    def __init__(self, cfg: PinnedConfig):
        self.cfg = cfg
        self.trigger_cell = np.full(cfg.n_vehicles, -1, np.int64)
        self.trigger_start = np.zeros(cfg.n_vehicles)

    def actions(self, runtime: PolicyRuntime, measured: np.ndarray,
                t_s: float) -> tuple[np.ndarray, np.ndarray]:
        nv, nc = measured.shape
        filtered = measured
        serving = measured[np.arange(nv), runtime.connected]
        eligible = filtered > (serving[:, None] + self.cfg.a3_offset_db +
                               self.cfg.a3_hysteresis_db)
        eligible[np.arange(nv), runtime.connected] = False
        candidates = np.where(eligible, filtered, -np.inf).argmax(axis=1)
        best_neighbor = filtered.copy()
        best_neighbor[np.arange(nv), runtime.connected] = -np.inf
        fallback = best_neighbor.argmax(axis=1)
        has = eligible.any(axis=1)
        actions = np.zeros(nv, np.int64)
        for v in range(nv):
            if not has[v]:
                self.trigger_cell[v] = -1
                candidates[v] = fallback[v]
                continue
            c = int(candidates[v])
            if self.trigger_cell[v] != c:
                self.trigger_cell[v] = c
                self.trigger_start[v] = t_s
            elif t_s - self.trigger_start[v] >= self.cfg.a3_ttt_s:
                actions[v] = 1
                self.trigger_cell[v] = -1
        return actions, candidates


def load_predictor(device: torch.device) -> tuple[nn.Module, dict, Path]:
    checkpoint = PREDICTOR_ROOT / "checkpoints" / "CNNLSTM_cell1.pth"
    scaler_path = PREDICTOR_ROOT / "data" / "processed" / "scaler.joblib"
    model = build_model("CNNLSTM", 12, 1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, joblib.load(scaler_path), checkpoint


def runtime_hardware() -> dict:
    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable"
    vram = None
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory
    return {
        "cpu_model": cpu_model,
        "logical_cores": os.cpu_count(),
        "ram_gib": round(mem_bytes / 2**30, 2),
        "gpu_model": gpu,
        "gpu_vram_gib": round(vram / 2**30, 2) if vram else None,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device_used": "GPU" if torch.cuda.is_available() else "CPU",
    }


def write_manifest(cfg: PinnedConfig, agent: DQNAgent, checkpoint: Path,
                   path: Path) -> None:
    cells = make_cells(cfg)
    manifest = {
        "environment": {
            "grid_m": [cfg.grid_width_m, cfg.grid_height_m],
            "cells_total": len(cells),
            "cell_distribution": {kind: sum(c.cell_type == kind for c in cells)
                                  for kind in ("macro", "micro", "pico")},
            "vehicles": cfg.n_vehicles,
            "dt_s": cfg.dt_s,
            "episode_steps": cfg.steps_per_episode,
            "episode_duration_s": cfg.dt_s * cfg.steps_per_episode,
            "vehicle_spawn_speed_kmh": [cfg.speed_spawn_min_kmh,
                                         cfg.speed_spawn_max_kmh],
            "vehicle_runtime_speed_clip_kmh": [cfg.speed_clip_min_kmh,
                                                 cfg.speed_clip_max_kmh],
            "mobility": "heading/speed random walk with boundary reflection",
            "interference": "distance-threshold overlap",
        },
        "evaluation": {
            "scenario_count": cfg.evaluation_scenarios,
            "scenario_reuse": True,
            "training_seed_count_per_learning_arm": 1,
            "training_episodes": cfg.training_episodes,
        },
        "agent": {
            "state_size": agent.q.layers[0].in_features,
            "action_size": agent.q.layers[-1].out_features,
            "hidden_widths": [agent.q.layers[i].out_features for i in (0, 4, 8, 12)],
            "dropout": [agent.q.layers[i].p for i in (3, 7, 11)],
            "optimizer": type(agent.optimizer).__name__,
            "learning_rate": agent.optimizer.param_groups[0]["lr"],
            "weight_decay": agent.optimizer.param_groups[0]["weight_decay"],
            "gamma": cfg.gamma,
            "epsilon_start": cfg.epsilon_start,
            "epsilon_min": cfg.epsilon_min,
            "epsilon_decay": cfg.epsilon_decay,
            "batch_size": cfg.batch_size,
            "replay_capacity": cfg.replay_capacity,
            "target_update_steps": cfg.target_update_steps,
            "warmup_updates": cfg.warmup_updates,
            "priority_alpha": cfg.priority_alpha,
            "importance_exponent": cfg.importance_exponent,
            "gradient_clip_norm": cfg.gradient_clip_norm,
            "algorithm": "Double DQN with prioritized experience replay",
            "scheduler": "Cosine annealing",
        },
        "constraint_layer": {
            "minimum_inter_handover_interval_s": cfg.min_handover_interval_s,
            "minimum_signal_improvement_db": cfg.min_rsrp_improvement_db,
            "blacklist_trigger_more_than_handover_count_per_300s": cfg.blacklist_count_300s,
            "blacklist_trigger_more_than_handover_count_per_60s": cfg.blacklist_count_60s,
            "blacklist_duration_s": cfg.blacklist_duration_s,
        },
        "baseline": {
            "offset_db": cfg.a3_offset_db,
            "hysteresis_db": cfg.a3_hysteresis_db,
            "time_to_trigger_s": cfg.a3_ttt_s,
            "radio_link_failure_threshold_dbm": cfg.rlf_threshold_dbm,
            "measurement_period_s": cfg.dt_s,
        },
        "predictor": {
            "display_architecture": "CNN-LSTM",
            "input_features": cfg.predictor_input_features,
            "history_steps": cfg.predictor_history,
            "output_steps": cfg.predictor_output_steps,
            "checkpoint_sha256": sha256_file(checkpoint),
            "normalization": "signal_norm=(signal_dbm+120)/60; fitted train-trace MinMax for other inputs",
            "truth_prediction_channel_shared_before_noise": True,
        },
        "reward": {
            "handover_cost": -2.0,
            "ping_pong_penalty": -20.0,
            "clip": [-30.0, 5.0],
            "implementation": "conditional signal, mobility, and stability terms",
        },
        "hardware": runtime_hardware(),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
