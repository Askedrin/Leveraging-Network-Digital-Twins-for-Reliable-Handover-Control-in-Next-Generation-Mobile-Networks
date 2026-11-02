#!/usr/bin/env python3
"""Phase 1b + Phase 2 confirmatory campaign (../PREREGISTRATION-2.md).

  python campaign.py mro                       warm up MRO-A3, select its step on calibration seeds
  python campaign.py train --seed 42           train the 4 learning arms, evaluate headline + sweeps
  python campaign.py eval --seed 42            (re-)evaluate saved agents only
  python campaign.py baselines                 seed-independent arms on every condition
  python campaign.py reference --seed 42       Phase 0b agents (paper's layer) on the new scenarios

Writes ../logs/raw-2/{seed-<s>,baselines,reference-<s>}/ ; aggregation: aggregate_campaign.py.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from forecaster_twin import DATA as FC_DATA, K, W_ERR, TwinForecaster
from run_seed import load_config
from twin import OUTPUT_ROOT, A3Controller, DQNAgent, Runtime, Scenario, runtime_hardware

ARMS = {"Reactive DQN": "reactive", "Oracle DQN k=5": "oracle", "Proactive DQN": "proactive",
        "Proactive DQN + fallback": "fallback"}
ARM_K = {"reactive": 0, "oracle": K, "proactive": K, "fallback": K}
HEADLINE = list(range(3000, 3050))
SWEEP = list(range(3000, 3020))
NOISE_SIGMAS = [1.0, 2.0, 3.0, 5.0, 8.0]
BIAS_DB = 3.0
MOBILITIES = ["random_walk", "random_waypoint", "gauss_markov"]
SPEEDS = [(10.0, 60.0), (30.0, 60.0), (100.0, 130.0)]
MRO_STEPS = [0.25, 0.5, 1.0]
MRO_WARMUP = range(9_000_000, 9_000_050)
RAW = OUTPUT_ROOT / "logs" / "raw-2"
CFG_DIR = OUTPUT_ROOT / "config-2"
STUDY = "2"


def configure_study(study: str):
    """Study 2: the pre-registered campaign. Study 3 (PREREGISTRATION-3.md): reliability-first
    use of forecasts, evaluated on fresh scenarios."""
    global RAW, HEADLINE, SWEEP, STUDY
    STUDY = study
    RAW = OUTPUT_ROOT / "logs" / f"raw-{study}"
    if study == "3":
        HEADLINE, SWEEP = list(range(3100, 3150)), list(range(3100, 3120))


def base_config():
    """Phase 0c environment and layer + the amended KPI-aligned reward (config-2/reward.json)."""
    cfg, baselines = load_config("config-0c")
    rw = CFG_DIR / "reward.json"
    if rw.exists():
        r = json.loads(rw.read_text())
        cfg = replace(cfg, reward_mode=r["reward_mode"], ho_cost=r["ho_cost"],
                      pp_penalty=r["pp_penalty"])
    if STUDY == "3":
        cfg = replace(cfg, danger_uses_measurement=True)
    return cfg, baselines


def conditions(cfg):
    """(condition, variant, cfg, scenario seeds, noise sigma, bias) for evaluation."""
    out = [("headline", "main", cfg, HEADLINE, 0.0, 0.0)]
    for s in NOISE_SIGMAS:
        out.append(("noise", f"sigma={s:g}", cfg, SWEEP, s, 0.0))
    out.append(("noise", f"bias=+{BIAS_DB:g}", cfg, SWEEP, 0.0, BIAS_DB))
    for m in MOBILITIES:
        out.append(("mobility", m, replace(cfg, mobility=m), SWEEP, 0.0, 0.0))
    for lo, hi in SPEEDS:
        out.append(("speed", f"{lo:g}-{hi:g}", replace(cfg, speed_min_kmh=lo, speed_max_kmh=hi),
                    SWEEP, 0.0, 0.0))
    return out


# ------------------------------------------------------------------------- runtime

class CampaignRuntime(Runtime):
    """Runtime whose signal source is the measurement, the oracle, or the basic model's
    forecast (optionally with the realised-error fallback)."""

    def __init__(self, cfg, sc, mode, fc=None, tau=np.inf, bias_db=0.0):
        super().__init__(cfg, sc)
        self.mode, self.fc, self.tau, self.bias = mode, fc, tau, bias_db
        self.issued: dict[int, np.ndarray] = {}
        self.per_err: list[np.ndarray] = []
        self.e = np.zeros(self.nv)
        self.fb_mask = np.zeros(self.nv, bool)
        self.danger_mask = np.zeros(self.nv, bool)
        self.c["fallback_steps"] = 0
        self.c["danger_steps"] = 0

    def source(self, t, k):
        sc = self.sc
        if self.mode == "reactive":
            return sc.meas[t]
        if self.mode == "oracle":
            return self._danger(t, sc.meas[t + K])
        idx = np.arange(self.nv)
        x = self.fc[t].astype(np.float64)
        if self.bias:
            x = x + self.bias
            x[idx, self.serving] -= self.bias
        self.issued[t] = x
        prev = self.issued.pop(t - K, None)
        if prev is not None:
            m = sc.meas[t]
            cand = m >= self.cfg.r_min_dbm
            self.per_err.append((np.abs(prev - m) * cand).sum(axis=1) /
                                np.maximum(cand.sum(axis=1), 1))
            if len(self.per_err) > W_ERR:
                self.per_err.pop(0)
            self.e = np.mean(self.per_err, axis=0)
        if self.mode == "fallback":
            self.fb_mask = self.e > self.tau
            if self.fb_mask.any():
                x = x.copy()
                x[self.fb_mask] = sc.meas[t][self.fb_mask]
            self.c["fallback_steps"] += int(self.fb_mask.sum())
        return self._danger(t, x)

    def _danger(self, t, x):
        """Study 3: a UE whose link is in danger uses its measurements (PREREGISTRATION-3.md)."""
        if not self.cfg.danger_uses_measurement:
            return x
        self.danger_mask = self.override(t)
        if self.danger_mask.any():
            x = np.array(x, dtype=np.float64, copy=True)
            x[self.danger_mask] = self.sc.meas[t][self.danger_mask]
        self.c["danger_steps"] += int(self.danger_mask.sum())
        return x

    def states(self, t, x, k):
        s, best = super().states(t, x, k)
        if self.mode in ("proactive", "fallback"):
            s[:, 12] = np.minimum(self.e / 10.0, 1.0)            # s13: realised forecast error
        meas_mask = self.fb_mask | self.danger_mask
        if self.mode != "reactive" and meas_mask.any():
            trend = np.tanh((self.hist[:, -1] - self.hist[:, -3]) / 10.0)
            s[meas_mask, 16] = trend[meas_mask]
        return s, best

    def metrics(self, arm, training_seed, scenario_seed):
        m = super().metrics(arm, training_seed, scenario_seed)
        m["fallback_frac"] = self.c["fallback_steps"] / max(self.c["ue_steps"], 1)
        m["danger_frac"] = self.c["danger_steps"] / max(self.c["ue_steps"], 1)
        return m


class MROController(A3Controller):
    """A3 with per cell-pair CIO adapted from TS 38.300 15.5.2 failure classes."""

    def __init__(self, cfg, margin_db, ttt_s, cio, delta, learn):
        super().__init__(cfg, margin_db, ttt_s)
        self.cio, self.delta, self.learn, self.seen = cio, delta, learn, 0

    def _update(self, rt):
        d, c = self.delta, self.cio
        for kind, a, b, m in rt.events[self.seen:]:
            if kind == "too_late":
                c[a, b] += d
            elif kind == "too_early" and a >= 0:
                c[a, b] -= d
            elif kind == "wrong_cell" and a >= 0:
                c[a, b] -= d
                c[a, m] += d
            elif kind == "ping_pong":
                c[a, b] -= d / 2
                c[b, a] -= d / 2
        np.clip(c, -6.0, 6.0, out=c)
        np.fill_diagonal(c, 0.0)
        self.seen = len(rt.events)

    def step(self, rt, t):
        rt.observe(t)
        sc, cfg = rt.sc, self.cfg
        t_s = (t + 1) * cfg.dt_s
        idx = np.arange(rt.nv)
        m = sc.meas[t] + self.cio[rt.serving]
        srv = sc.meas[t][idx, rt.serving]
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
        if self.learn:
            self._update(rt)


# --------------------------------------------------------------------------- helpers

def forecast_tools():
    sel = json.loads((FC_DATA / "selection.json").read_text())
    return TwinForecaster(sel["selected_arch"]), float(sel["tau_e_db"]), sel


def noisy(fc, scenario_seed, sigma):
    if sigma <= 0:
        return fc
    rng = np.random.default_rng([scenario_seed, int(round(sigma * 100)), 7])
    return (fc + rng.normal(0.0, sigma, fc.shape)).astype(np.float32)


def run_agent_arm(cfg, sc, arm, agent, fc, tau, bias, seed, scenario_seed):
    rt = CampaignRuntime(cfg, sc, ARMS[arm], fc, tau, bias)
    for t in range(cfg.steps_per_episode):
        rt.step_agent(agent, t, ARM_K[ARMS[arm]], training=False)
    return rt.metrics(arm, seed, scenario_seed)


def run_rules(cfg, sc, arm, scenario_seed):
    rt = Runtime(cfg, sc)
    for t in range(cfg.steps_per_episode):
        rt.step_rules(t)
    return rt.metrics(arm, "n/a", scenario_seed)


def run_a3(cfg, sc, arm, margin, ttt, scenario_seed, cio=None, delta=0.0, learn=False):
    rt = Runtime(cfg, sc)
    ctl = (A3Controller(cfg, margin, ttt) if cio is None
           else MROController(cfg, margin, ttt, cio, delta, learn))
    for t in range(cfg.steps_per_episode):
        ctl.step(rt, t)
    return rt.metrics(arm, "n/a", scenario_seed)


def paper_layer(cfg):
    return replace(cfg, reliability_override=False, admission_check=False)


# ----------------------------------------------------------------------------- jobs

def train(cfg, seed, episodes, fcm, tau, out):
    device = torch.device("cpu")
    agents = {arm: DQNAgent(cfg, seed, device) for arm in ARMS}
    rows, start = [], time.perf_counter()
    for ep in range(episodes):
        t0 = time.perf_counter()
        sc_seed = seed * 1_000_000 + ep
        sc = Scenario(cfg, sc_seed)
        fc = fcm.predict(sc)
        rts = {arm: CampaignRuntime(cfg, sc, ARMS[arm], fc, tau) for arm in agents}
        for t in range(cfg.steps_per_episode):
            rng_before, rng_after = torch.get_rng_state(), None
            for i, (arm, agent) in enumerate(agents.items()):
                if i:
                    torch.set_rng_state(rng_before)
                rts[arm].step_agent(agent, t, ARM_K[ARMS[arm]], training=True)
                agent.train_update()
                if i == 0:
                    rng_after = torch.get_rng_state()
            torch.set_rng_state(rng_after)
        for arm, agent in agents.items():
            m = rts[arm].metrics(arm, seed, sc_seed)
            rows.append({"arm": arm, "training_seed": seed, "episode": ep + 1,
                         "scenario_seed": sc_seed, "mean_episode_reward": m["mean_reward"],
                         "epsilon": agent.epsilon, "n_ho": m["n_ho"], "n_hof": m["n_hof"],
                         "n_rlf": m["n_rlf"], "outage_frac": m["outage_frac"],
                         "fallback_frac": m["fallback_frac"]})
        print(f"seed {seed} ep {ep + 1}/{episodes} {time.perf_counter() - t0:.1f}s | " +
              " | ".join(f"{r['arm']}: R={r['mean_episode_reward']:.3f} ho={r['n_ho']} "
                         f"rlf={r['n_rlf']}" for r in rows[-len(agents):]), flush=True)
    pd.DataFrame(rows).to_csv(out / "reward_curves.csv", index=False)
    for arm, agent in agents.items():
        agent.save(out / f"agent-{ARMS[arm]}.pth")
    return agents, time.perf_counter() - start


def load_agents(cfg, seed, out):
    agents = {}
    for arm, mode in ARMS.items():
        a = DQNAgent(cfg, seed, torch.device("cpu"))
        a.load(out / f"agent-{mode}.pth")
        agents[arm] = a
    return agents


def evaluate(cfg, seed, agents, fcm, tau, out):
    rows, start = [], time.perf_counter()
    cache = {}                       # main-configuration scenarios and forecasts, reused by noise
    for cond, var, ccfg, seeds, sigma, bias in conditions(cfg):
        arms = ["Proactive DQN", "Proactive DQN + fallback"] if cond == "noise" else list(ARMS)
        for s in seeds:
            if ccfg == cfg and s in cache:
                sc, fc0 = cache[s]
            else:
                sc = Scenario(ccfg, s)
                fc0 = fcm.predict(sc)
                if ccfg == cfg and s in SWEEP:
                    cache[s] = (sc, fc0)
            fc = noisy(fc0, s, sigma)
            for arm in arms:
                m = run_agent_arm(ccfg, sc, arm, agents[arm], fc, tau, bias, seed, s)
                rows.append({"condition": cond, "variant": var, **m})
        print(f"seed {seed} eval {cond}/{var} done {time.perf_counter() - start:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "eval.csv", index=False)
    return time.perf_counter() - start


def baselines_job(cfg, baselines):
    out = RAW / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    mro = json.loads((CFG_DIR / "mro.json").read_text())
    cio0 = np.array(mro["cio"])
    a3s, a3t = baselines["static_a3"], baselines["tuned_a3"]
    rows, start = [], time.perf_counter()
    for cond, var, ccfg, seeds, sigma, bias in conditions(cfg):
        if cond == "noise":
            continue
        for s in seeds:
            sc = Scenario(ccfg, s)
            rows.append({"condition": cond, "variant": var,
                         **run_a3(ccfg, sc, "Static A3", a3s["margin_db"], a3s["ttt_s"], s)})
            rows.append({"condition": cond, "variant": var,
                         **run_a3(ccfg, sc, "Tuned A3", a3t["margin_db"], a3t["ttt_s"], s)})
            rows.append({"condition": cond, "variant": var,
                         **run_a3(ccfg, sc, "MRO-A3", a3s["margin_db"], a3s["ttt_s"], s,
                                  cio=cio0.copy(), delta=0.0, learn=False)})
            rows.append({"condition": cond, "variant": var, **run_rules(ccfg, sc, "Rules only", s)})
            rows.append({"condition": cond, "variant": var,
                         **run_rules(paper_layer(ccfg), sc, "Rules only, paper's layer", s)})
        print(f"baselines {cond}/{var} done {time.perf_counter() - start:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "eval.csv", index=False)


REFERENCES = {"0b": "paper's layer (Phase 0b)", "0c": "original reward (Phase 0c)"}


def reference_job(seed, phase):
    """Earlier-phase agents (original reward; 0b: paper's layer, 0c: reliability-aware layer)
    on the new headline scenarios, in their own configuration."""
    from twin import LEARNING_ARMS
    cfg0, _ = load_config(f"config-{phase}")
    out = RAW / f"reference-{phase}-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    src = OUTPUT_ROOT / "logs" / f"raw-{phase}" / f"seed-{seed}"
    rows = []
    for arm, fname in (("Reactive DQN", "agent-Reactive_DQN.pth"),
                       ("Oracle DQN k=5", "agent-Oracle_DQN_k5.pth")):
        agent = DQNAgent(cfg0, seed, torch.device("cpu"))
        agent.load(src / fname)
        for s in HEADLINE:
            sc = Scenario(cfg0, s)
            rt = Runtime(cfg0, sc)
            for t in range(cfg0.steps_per_episode):
                rt.step_agent(agent, t, LEARNING_ARMS[arm], training=False)
            rows.append({"condition": "headline", "variant": "main",
                         **rt.metrics(f"{arm}, {REFERENCES[phase]}", seed, s)})
    pd.DataFrame(rows).to_csv(out / "eval.csv", index=False)


def prereg_reference_job(seed, tau):
    """Study 3: the campaign's forecast-based agents (no danger rule) on the Study-3 headline."""
    cfg, _ = base_config()
    cfg2 = replace(cfg, danger_uses_measurement=False)
    agents = load_agents(cfg2, seed, OUTPUT_ROOT / "logs" / "raw-2" / f"seed-{seed}")
    fcm = forecast_tools()[0]
    out = RAW / f"prereg-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in HEADLINE:
        sc = Scenario(cfg2, s)
        fc = fcm.predict(sc)
        for arm in ("Oracle DQN k=5", "Proactive DQN", "Proactive DQN + fallback"):
            m = run_agent_arm(cfg2, sc, arm, agents[arm], fc, tau, 0.0, seed, s)
            m["arm"] = f"{arm}, pre-registered design"
            rows.append({"condition": "headline", "variant": "main", **m})
    pd.DataFrame(rows).to_csv(out / "eval.csv", index=False)


SENS_DIR = FC_DATA / "transformer-forecasts"


def sens_forecasts_job(cfg, part=0, parts=1):
    """DEVIATIONS.md D1: cache the Transformer's forecasts for the headline scenarios."""
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    fcm = TwinForecaster("Transformer")
    for s in HEADLINE[part::parts]:
        f = SENS_DIR / f"{s}.npy"
        if not f.exists():
            np.save(f, fcm.predict(Scenario(cfg, s)))
            print("forecast", s, flush=True)


def sensitivity_job(cfg, seed, tau):
    """Proactive arms of one seed on the headline scenarios with the Transformer's forecasts."""
    agents = load_agents(cfg, seed, RAW / f"seed-{seed}")
    out = RAW / f"sensitivity-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in HEADLINE:
        sc = Scenario(cfg, s)
        fc = np.load(SENS_DIR / f"{s}.npy")
        for arm in ("Proactive DQN", "Proactive DQN + fallback"):
            m = run_agent_arm(cfg, sc, arm, agents[arm], fc, tau, 0.0, seed, s)
            rows.append({"condition": "sensitivity", "variant": "transformer-forecasts", **m})
    pd.DataFrame(rows).to_csv(out / "eval.csv", index=False)


def mro_job(cfg, baselines):
    """Warm up the CIOs on MRO_WARMUP scenarios, select the step on calibration seeds."""
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    a3s = baselines["static_a3"]
    nc = len(Scenario(cfg, MRO_WARMUP[0]).cells)
    warm = [Scenario(cfg, s) for s in MRO_WARMUP]
    calib = [Scenario(cfg, s) for s in cfg.calibration_seeds]
    grid, cios = [], {}
    for delta in MRO_STEPS:
        cio = np.zeros((nc, nc))
        for s, sc in zip(MRO_WARMUP, warm):
            run_a3(cfg, sc, "MRO-A3", a3s["margin_db"], a3s["ttt_s"], s, cio=cio, delta=delta,
                   learn=True)
        df = pd.DataFrame([run_a3(cfg, sc, "MRO-A3", a3s["margin_db"], a3s["ttt_s"], s,
                                  cio=cio.copy(), delta=0.0, learn=False)
                           for s, sc in zip(cfg.calibration_seeds, calib)])
        ueh = df.ue_steps.sum() * cfg.dt_s / 3600
        grid.append({"delta_db": delta,
                     "objective_per_ueh": (df.n_rlf.sum() + df.n_hof.sum() +
                                           0.1 * df.n_pp1.sum()) / ueh,
                     "ho_per_ueh": df.n_ho.sum() / ueh, "rlf_per_ueh": df.n_rlf.sum() / ueh,
                     "ppr1": df.n_pp1.sum() / max(df.n_ho.sum(), 1),
                     "cio_mean_abs_db": float(np.abs(cio).mean()),
                     "cio_max_abs_db": float(np.abs(cio).max())})
        cios[delta] = cio
        print(json.dumps(grid[-1]), flush=True)
    g = pd.DataFrame(grid).sort_values(["objective_per_ueh", "ho_per_ueh"])
    g.to_csv(CFG_DIR / "mro-grid.csv", index=False)
    best = float(g.iloc[0].delta_db)
    (CFG_DIR / "mro.json").write_text(json.dumps({
        "delta_db": best, "base_margin_db": a3s["margin_db"], "ttt_s": a3s["ttt_s"],
        "warmup_scenarios": [MRO_WARMUP.start, MRO_WARMUP.stop - 1],
        "selection": "min (N_RLF + N_HOF + 0.1 N_PP,1s) per UE-h on calibration seeds",
        "cio": cios[best].round(4).tolist()}, indent=2) + "\n")


def write_pinned(cfg, baselines, sel):
    out = OUTPUT_ROOT / f"config-{STUDY}"
    out.mkdir(parents=True, exist_ok=True)
    pinned = asdict(cfg)
    pinned["hardware"] = runtime_hardware()
    pinned["campaign"] = {"arms": ARMS, "seeds": [42, 43, 44, 45, 46], "headline": HEADLINE,
                          "sweep": SWEEP, "noise_sigmas": NOISE_SIGMAS, "bias_db": BIAS_DB,
                          "mobilities": MOBILITIES, "speeds": SPEEDS, "forecaster": sel,
                          "baselines": baselines}
    (out / "pinned-config.json").write_text(json.dumps(pinned, indent=2, default=str) + "\n")
    (out / "baselines.json").write_text(json.dumps(baselines, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mro", "train", "eval", "baselines", "reference",
                                    "sens-forecasts", "sensitivity", "prereg-ref"])
    ap.add_argument("--study", choices=["2", "3"], default="2")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--phase", choices=list(REFERENCES), default="0b")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    configure_study(a.study)
    cfg, baselines = base_config()
    if a.cmd == "mro":
        mro_job(cfg, baselines)
        return
    if a.cmd == "baselines":
        baselines_job(cfg, baselines)
        return
    if a.cmd == "reference":
        reference_job(a.seed, a.phase)
        return
    if a.cmd == "sens-forecasts":
        sens_forecasts_job(cfg, a.part, a.parts)
        return
    fcm, tau, sel = forecast_tools()
    if a.cmd == "sensitivity":
        sensitivity_job(cfg, a.seed, tau)
        return
    if a.cmd == "prereg-ref":
        prereg_reference_job(a.seed, tau)
        return
    write_pinned(cfg, baselines, sel)
    out = RAW / f"seed-{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    train_s = None
    if a.cmd == "train":
        agents, train_s = train(cfg, a.seed, a.episodes or cfg.training_episodes, fcm, tau, out)
    else:
        agents = load_agents(cfg, a.seed, out)
    eval_s = evaluate(cfg, a.seed, agents, fcm, tau, out)
    pd.DataFrame([{"training_seed": a.seed, "training_runtime_s": train_s,
                   "evaluation_runtime_s": round(eval_s, 1), "threads": a.threads,
                   "forecaster": sel["selected_arch"], "tau_e_db": tau}]).to_csv(
        out / f"run-summary-{a.cmd}.csv", index=False)


if __name__ == "__main__":
    main()
