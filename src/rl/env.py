"""Среда для DQN-тренировки RLTunedGLFTStrategy. ABIDES callback-driven,
поэтому работаем offline: `run_episode` запускает одну симуляцию, после
чего из `strategy.history_log` собираются `(s, a, r, s', done)` переходы
для replay buffer'а. Награда: Δequity − λ·q² (с делителем 1000 для шкалы O(1))."""

import io
import contextlib
import numpy as np
import pandas as pd

from Kernel import Kernel

from src.scenarios import multimarket_baseline as mmb
from src.agents.unified_mm import UnifiedMM
from src.agents.mm_strategies import RLTunedGLFTStrategy


# Дефолтные kwargs для `mmb.build()` — совпадают с `comparison.ipynb`'s
# `COMMON_ARGS`. Это «лёгкий» сетап для быстрой RL-тренировки.
DEFAULT_BUILD_KWARGS = dict(
    num_noise=500,
    num_value=25,
    num_momentum=5,
    num_zi=50,
    value_lambda_a_override=None,
    value_sigma_n_override=None,
    mm_wake_freq_override=None,
    num_noise_fx=3000,
    num_zi_fx=200,
    num_momentum_fx=10,
    arb_order_size=1,
    arb_open_pct=0.3,
    arb_close_pct=0.15,
    arb_wake_freq="10s",
    arb_starting_inventory=0,
    arb_max_exposure=1000,
    mm_starting_cash=50_000_000,
    arb_starting_cash=20_000_000,
    maker_rebate=0.0,
    mm_type="rl_glft",
    # AS / GLFT defaults — нерелевантны для rl_glft, но build() требует значения.
    as_gamma=5e-5, as_k=1.5, as_sigma_a=4.4, as_sigma_b=3.2,
    glft_gamma=5e-7, glft_k=1.5, glft_A=1.0, glft_sigma=4.4,
    rl_gamma_set=(1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5),
    rl_sigma=4.4, rl_k=1.5, rl_A=1.0,
    rl_policy_path=None,
    sb_order_size=1, sb_window_size=5, sb_num_ticks=20, sb_anchor="bottom",
    adp_min_order_size=20, adp_window_size="2", adp_num_ticks=20,
    adp_anchor="middle", adp_skew_beta=10.0,
    fast_mode=True,
    stress_multiplier=1.0,
    megashock_mean=10.0,
    megashock_var=5.0,
    fx_shock_specs=(),
    news_shock_specs=(),
    drift_shock_specs=(),
    liquidity_shock_specs=(),
    pressure_shock_specs=(),
)


# Те же 3 сценария, что в `comparison.ipynb` (тренировочный subset).
SCENARIO_PRESETS = {
    "stationary": {},
    "megashock":  {"stress_multiplier": 1e6, "megashock_mean": 100.0},
    "drift_down": {"drift_shock_specs": ("10:00-10:15:-500",)},
}


def _build_kwargs(seed, scenario, overrides=None):
    kw = dict(DEFAULT_BUILD_KWARGS)
    kw["seed"] = seed
    kw.update(SCENARIO_PRESETS.get(scenario, {}))
    if overrides:
        kw.update(overrides)
    return kw


def _find_rl_strategies(agents):
    return [(a.name, a.strategy) for a in agents
            if isinstance(a, UnifiedMM) and isinstance(a.strategy, RLTunedGLFTStrategy)]


def _trajectory_from_log(history_log, lambda_inv):
    # Reward: (Δequity − λ·q²) / 1000. При len<2 возвращаем пустые массивы.
    n = len(history_log)
    if n < 2:
        return {
            "states": np.zeros((0, RLTunedGLFTStrategy.STATE_DIM), dtype=np.float32),
            "actions": np.zeros((0,), dtype=np.int64),
            "rewards": np.zeros((0,), dtype=np.float32),
            "next_states": np.zeros((0, RLTunedGLFTStrategy.STATE_DIM), dtype=np.float32),
            "dones": np.zeros((0,), dtype=np.float32),
        }
    states = np.stack([h["state"] for h in history_log], axis=0)
    actions = np.array([h["action"] for h in history_log], dtype=np.int64)
    equities = np.array(
        [h["equity"] if h["equity"] is not None else 0.0 for h in history_log],
        dtype=np.float64,
    )
    inventories = np.array([h["inventory"] for h in history_log], dtype=np.float64)

    delta_eq = (equities[1:] - equities[:-1]) / 1000.0
    inv_pen  = lambda_inv * (inventories[:-1] ** 2) / 1000.0
    rewards  = (delta_eq - inv_pen).astype(np.float32)

    dones = np.zeros(n - 1, dtype=np.float32)
    dones[-1] = 1.0   # последний переход — terminal

    return {
        "states": states[:-1].astype(np.float32),
        "actions": actions[:-1],
        "rewards": rewards,
        "next_states": states[1:].astype(np.float32),
        "dones": dones,
    }


def run_episode(seed, scenario, policy_fn, lambda_inv=0.05,
                build_overrides=None, suppress_stdout=True):
    """Один эпизод: запуск симуляции, сбор trajectory и диагностики.
    `policy_fn=None` → fallback на action=0 (нужно до старта тренировки)."""
    kw = _build_kwargs(seed, scenario, build_overrides)
    agents, oracle, kernel_start, mkt_close, kernel_seed = mmb.build(**kw)

    rl_strategies = _find_rl_strategies(agents)
    if not rl_strategies:
        raise RuntimeError("run_episode: RLTunedGLFTStrategy не найден в агентах")

    # Reset trajectory log и подключаем policy в обе RL-стратегии.
    for _name, strat in rl_strategies:
        strat.reset_history()
        strat.set_policy(policy_fn)

    kernel = Kernel("RL Training Kernel",
                    random_state=np.random.RandomState(seed=kernel_seed))
    runner_kwargs = dict(
        agents=agents,
        startTime=kernel_start,
        stopTime=mkt_close + pd.to_timedelta("00:01:00"),
        defaultComputationDelay=50,
        defaultLatency=50,
        oracle=oracle,
        log_dir=None,    # RL training не пишет на диск
    )
    if suppress_stdout:
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.runner(**runner_kwargs)
    else:
        kernel.runner(**runner_kwargs)

    # Concat trajectory из обеих RL-стратегий (A_USD + B_EUR).
    all_states, all_actions, all_rewards, all_next, all_dones = [], [], [], [], []
    all_inventories = []   # все wakeups (не только переходы) — для |q|-stats
    final_equity_total = 0.0
    initial_equity_total = 0.0
    n_actions = len(rl_strategies[0][1].gamma_set)
    for _name, strat in rl_strategies:
        traj = _trajectory_from_log(strat.history_log, lambda_inv)
        all_states.append(traj["states"])
        all_actions.append(traj["actions"])
        all_rewards.append(traj["rewards"])
        all_next.append(traj["next_states"])
        all_dones.append(traj["dones"])
        for h in strat.history_log:
            all_inventories.append(h["inventory"])
        if strat.history_log:
            first, last = strat.history_log[0], strat.history_log[-1]
            if first["equity"] is not None:
                initial_equity_total += first["equity"]
            if last["equity"] is not None:
                final_equity_total += last["equity"]

    trajectory = {
        "states": np.concatenate(all_states, axis=0)
                  if all_states else np.zeros((0, RLTunedGLFTStrategy.STATE_DIM), dtype=np.float32),
        "actions": np.concatenate(all_actions, axis=0)
                   if all_actions else np.zeros((0,), dtype=np.int64),
        "rewards": np.concatenate(all_rewards, axis=0)
                   if all_rewards else np.zeros((0,), dtype=np.float32),
        "next_states": np.concatenate(all_next, axis=0)
                       if all_next else np.zeros((0, RLTunedGLFTStrategy.STATE_DIM), dtype=np.float32),
        "dones": np.concatenate(all_dones, axis=0)
                 if all_dones else np.zeros((0,), dtype=np.float32),
    }

    n_steps = trajectory["states"].shape[0]
    if n_steps > 0:
        action_hist = np.bincount(trajectory["actions"], minlength=n_actions).tolist()
        mean_action = float(trajectory["actions"].mean())
    else:
        action_hist = [0] * n_actions
        mean_action = 0.0
    inv_arr = np.abs(np.asarray(all_inventories, dtype=np.float64)) if all_inventories else np.zeros(0)
    return {
        "trajectory": trajectory,
        "episode_return": float(trajectory["rewards"].sum()),
        "mean_action": mean_action,
        "action_hist": action_hist,
        "n_steps": n_steps,
        "mean_abs_inventory": float(inv_arr.mean()) if inv_arr.size else 0.0,
        "max_abs_inventory": int(inv_arr.max()) if inv_arr.size else 0,
        "final_equity_total": final_equity_total,
        "equity_pnl": final_equity_total - initial_equity_total,
    }


class ReplayBuffer:
    """Circular buffer на numpy-массивах. `push(traj)` добавляет переходы из
    `run_episode`, `sample(n)` — равномерно (без приоритетов)."""

    def __init__(self, capacity, state_dim):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.size = 0
        self.idx = 0    # позиция для записи следующего перехода
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, trajectory):
        n = trajectory["states"].shape[0]
        if n == 0:
            return
        # Circular append: 1 или 2 отрезка относительно self.idx.
        end = self.idx + n
        if end <= self.capacity:
            sl = slice(self.idx, end)
            self.states[sl] = trajectory["states"]
            self.actions[sl] = trajectory["actions"]
            self.rewards[sl] = trajectory["rewards"]
            self.next_states[sl] = trajectory["next_states"]
            self.dones[sl] = trajectory["dones"]
        else:
            first = self.capacity - self.idx
            second = n - first
            self.states[self.idx:] = trajectory["states"][:first]
            self.actions[self.idx:] = trajectory["actions"][:first]
            self.rewards[self.idx:] = trajectory["rewards"][:first]
            self.next_states[self.idx:] = trajectory["next_states"][:first]
            self.dones[self.idx:] = trajectory["dones"][:first]
            self.states[:second] = trajectory["states"][first:]
            self.actions[:second] = trajectory["actions"][first:]
            self.rewards[:second] = trajectory["rewards"][first:]
            self.next_states[:second] = trajectory["next_states"][first:]
            self.dones[:second] = trajectory["dones"][first:]
        self.idx = (self.idx + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, random_state=None):
        rs = random_state if random_state is not None else np.random
        idx = rs.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }

    def __len__(self):
        return self.size

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "size": self.size,
            "idx": self.idx,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
        }

    def load_state_dict(self, sd):
        if sd["capacity"] != self.capacity or sd["state_dim"] != self.state_dim:
            raise ValueError(
                f"buffer shape mismatch: ckpt={sd['capacity']}x{sd['state_dim']}, "
                f"current={self.capacity}x{self.state_dim}"
            )
        self.size = int(sd["size"])
        self.idx = int(sd["idx"])
        self.states[:] = sd["states"]
        self.actions[:] = sd["actions"]
        self.rewards[:] = sd["rewards"]
        self.next_states[:] = sd["next_states"]
        self.dones[:] = sd["dones"]
