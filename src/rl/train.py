"""Тренировка DQN-policy для RLTunedGLFTStrategy. Curriculum в 3 фазы:
stationary → +drift_down → +megashock. Артефакты в `experiments/rl/dqn_run_<ts>/`
(`config.json`, `train_log.jsonl`, `eval_log.jsonl`, `last.pt` для resume,
`best.pt` по лучшему eval, периодические checkpoints).

CLI: `python -m src.rl.train [--n-episodes 500] [--resume <out_dir>]`."""

import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.agents.mm_strategies import RLTunedGLFTStrategy
from src.rl.dqn import DQNAgent
from src.rl.env import ReplayBuffer, run_episode


def curriculum_scenario(episode_idx, n_total, rng):
    # 0–40%: stationary; 40–70%: +drift_down; 70–100%: +megashock.
    p1 = int(0.4 * n_total)
    p2 = int(0.7 * n_total)
    if episode_idx < p1:
        return "stationary"
    if episode_idx < p2:
        return rng.choice(["stationary", "drift_down"])
    return rng.choice(["stationary", "drift_down", "megashock"])


EVAL_SCENARIOS = ["stationary", "drift_down", "megashock"]
EVAL_SEEDS = [100, 101, 102]


def evaluate(agent, lambda_inv, build_overrides=None):
    # Greedy eval: один фикс-сидный эпизод на сценарий.
    policy = agent.make_greedy_policy()
    per_scenario = {}
    returns = []
    for scenario, seed in zip(EVAL_SCENARIOS, EVAL_SEEDS):
        res = run_episode(seed=seed, scenario=scenario, policy_fn=policy,
                          lambda_inv=lambda_inv, build_overrides=build_overrides)
        per_scenario[scenario] = {
            "return": res["episode_return"],
            "mean_abs_inventory": res["mean_abs_inventory"],
            "max_abs_inventory": res["max_abs_inventory"],
            "equity_pnl_cents": res["equity_pnl"],
            "action_hist": res["action_hist"],
        }
        returns.append(res["episode_return"])
    return {
        "mean_return": float(np.mean(returns)),
        "per_scenario": per_scenario,
    }


def _format_action_hist(hist, gamma_set, width=40):
    total = sum(hist) or 1
    lines = []
    max_pct = max(hist) / total if total else 0
    for i, (g, c) in enumerate(zip(gamma_set, hist)):
        pct = c / total
        bar_len = int(width * pct / max(max_pct, 1e-9))
        lines.append(f"    γ={g:.0e}  {('#' * bar_len).ljust(width)} {100*pct:5.1f}%")
    return "\n".join(lines)


def train(
    n_episodes=500,
    lambda_inv=0.05,
    gamma_set=(1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5),
    hidden=64,
    lr=3e-4,
    gamma_disc=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=300,
    buffer_capacity=50_000,
    batch_size=64,
    grad_steps_per_episode=50,
    warmup_episodes=5,
    eval_every=25,
    checkpoint_every=10,
    summary_every=25,
    seed=0,
    out_dir=None,
    resume_from=None,
    # GLFT/env параметры — пробрасываются в run_episode через build_overrides.
    # Влияют на η/ψ; изменение между тренировкой и inference = out-of-distribution
    # policy. Сохраняются в config.json, чтобы resume и inference использовали
    # тот же режим.
    rl_A=1.0,
    rl_sigma=4.4,
    rl_k=1.5,
):
    """При `resume_from` все non-default аргументы игнорируются — берутся
    из `config.json`."""
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_from не существует: {resume_path}")
        out_dir = resume_path
        config_path = out_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json не найден в {out_dir}")
        config = json.loads(config_path.read_text())
        print(f"[resume] Загружаю config из {config_path}")
        # Все гиперпараметры берутся из config; переданные kwargs игнорируются.
        n_episodes = config["n_episodes"]
        lambda_inv = config["lambda_inv"]
        gamma_set = tuple(config["gamma_set"])
        hidden = config["hidden"]
        lr = config["lr"]
        gamma_disc = config["gamma_disc"]
        tau = config["tau"]
        eps_start = config["eps_start"]
        eps_end = config["eps_end"]
        eps_decay_episodes = config["eps_decay_episodes"]
        buffer_capacity = config["buffer_capacity"]
        batch_size = config["batch_size"]
        grad_steps_per_episode = config["grad_steps_per_episode"]
        warmup_episodes = config["warmup_episodes"]
        eval_every = config["eval_every"]
        checkpoint_every = config["checkpoint_every"]
        summary_every = config.get("summary_every", 25)
        seed = config["seed"]
        rl_A = config.get("rl_A", 1.0)
        rl_sigma = config.get("rl_sigma", 4.4)
        rl_k = config.get("rl_k", 1.5)
    else:
        if out_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(__file__).resolve().parents[2] / "experiments" / "rl" / f"dqn_run_{ts}"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)

    state_dim = RLTunedGLFTStrategy.STATE_DIM
    n_actions = len(gamma_set)
    eps_decay_steps = eps_decay_episodes * grad_steps_per_episode

    # Сохраняем config только при fresh-start (на resume он уже прочитан выше).
    if resume_from is None:
        config = dict(
            n_episodes=n_episodes, lambda_inv=lambda_inv, gamma_set=list(gamma_set),
            hidden=hidden, lr=lr, gamma_disc=gamma_disc, tau=tau,
            eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay_episodes,
            eps_decay_steps=eps_decay_steps,
            buffer_capacity=buffer_capacity, batch_size=batch_size,
            grad_steps_per_episode=grad_steps_per_episode,
            warmup_episodes=warmup_episodes,
            eval_every=eval_every, checkpoint_every=checkpoint_every,
            summary_every=summary_every,
            seed=seed, state_dim=state_dim, n_actions=n_actions,
            state_features=RLTunedGLFTStrategy.STATE_FEATURES,
            rl_A=rl_A, rl_sigma=rl_sigma, rl_k=rl_k,
        )
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # build_overrides пробрасываются и в training-run_episode, и в evaluate —
    # гарантируем единый η/ψ режим стратегии.
    build_overrides = {"rl_A": rl_A, "rl_sigma": rl_sigma, "rl_k": rl_k}

    agent = DQNAgent(
        state_dim=state_dim, n_actions=n_actions, gamma_set=gamma_set,
        hidden=hidden, gamma_disc=gamma_disc, lr=lr, tau=tau,
        eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay_steps,
        device="cpu", seed=seed,
    )
    buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim)
    train_rng = np.random.RandomState(seed)

    start_ep = 0
    best_eval = -float("inf")

    last_pt = out_dir / "last.pt"
    if resume_from is not None and last_pt.exists():
        ckpt = agent.load_training_state(last_pt)
        if "buffer_state" in ckpt:
            buffer.load_state_dict(ckpt["buffer_state"])
        if "train_rng_state" in ckpt:
            train_rng.set_state(ckpt["train_rng_state"])
        start_ep = int(ckpt.get("episode", 0))
        best_eval = float(ckpt.get("best_eval", -float("inf")))
        print(f"[resume] state восстановлен: episode={start_ep}, "
              f"buffer={len(buffer)}, step_count={agent.step_count}, "
              f"best_eval={best_eval:+.2f}, eps={agent.epsilon:.3f}")
    elif resume_from is not None:
        print(f"[resume] warning: {last_pt} не найден, начинаю с нуля")

    train_log_path = out_dir / "train_log.jsonl"
    eval_log_path  = out_dir / "eval_log.jsonl"
    if start_ep == 0:
        train_log_path.write_text("")
        eval_log_path.write_text("")

    # Скользящие окна для console summary (rolling-mean returns, action histogram).
    recent_returns     = collections.deque(maxlen=summary_every)
    recent_action_hist = np.zeros(n_actions, dtype=np.int64)
    recent_abs_inv     = collections.deque(maxlen=summary_every)

    t0 = time.time()

    def _save_last(ep_done):
        agent.save_training_state(
            last_pt,
            episode=ep_done,
            best_eval=best_eval,
            buffer_state=buffer.state_dict(),
            train_rng_state=train_rng.get_state(),
        )

    try:
        for ep in range(start_ep, n_episodes):
            scenario = curriculum_scenario(ep, n_episodes, train_rng)

            # Warmup-эпизоды: чисто uniform policy, чтобы наполнить буфер
            # разнообразными переходами до того, как DQN начнёт учиться.
            if ep < warmup_episodes:
                policy = lambda s, _n=n_actions: int(train_rng.randint(0, _n))
            else:
                policy = agent.make_eps_greedy_policy()

            episode_seed = int(train_rng.randint(0, 2**31 - 1))
            res = run_episode(seed=episode_seed, scenario=scenario,
                              policy_fn=policy, lambda_inv=lambda_inv,
                              build_overrides=build_overrides)
            buffer.push(res["trajectory"])

            # Gradient steps только после warmup'а и при достаточном буфере.
            train_metrics = {"loss": None, "mean_q": None, "epsilon": agent.epsilon}
            if len(buffer) >= batch_size and ep >= warmup_episodes:
                losses, qs = [], []
                for _ in range(grad_steps_per_episode):
                    m = agent.train_step(buffer.sample(batch_size, random_state=agent.rng))
                    losses.append(m["loss"])
                    qs.append(m["mean_q"])
                train_metrics = {"loss": float(np.mean(losses)),
                                 "mean_q": float(np.mean(qs)),
                                 "epsilon": agent.epsilon}

            elapsed = time.time() - t0
            line = {
                "ep": ep,
                "scenario": scenario,
                "episode_return": res["episode_return"],
                "n_steps": res["n_steps"],
                "mean_action": res["mean_action"],
                "action_hist": res["action_hist"],
                "mean_abs_inventory": res["mean_abs_inventory"],
                "max_abs_inventory": res["max_abs_inventory"],
                "equity_pnl_cents": res["equity_pnl"],
                "buffer_size": len(buffer),
                "elapsed_sec": elapsed,
                **train_metrics,
            }
            with train_log_path.open("a") as f:
                f.write(json.dumps(line) + "\n")

            recent_returns.append(res["episode_return"])
            recent_action_hist += np.asarray(res["action_hist"], dtype=np.int64)
            recent_abs_inv.append(res["mean_abs_inventory"])

            loss_str = (f"{train_metrics['loss']:7.3f}"
                        if train_metrics["loss"] is not None else "  n/a  ")
            print(f"ep={ep:4d} sc={scenario:11s} R={res['episode_return']:+8.2f} "
                  f"n={res['n_steps']:4d} |q|̄={res['mean_abs_inventory']:5.1f} "
                  f"eps={agent.epsilon:.3f} loss={loss_str} "
                  f"buf={len(buffer):5d} t={elapsed:.0f}s")

            # Full state каждый эпизод — для seamless resume.
            _save_last(ep + 1)

            # Inference-only ckpt каждые checkpoint_every.
            if (ep + 1) % checkpoint_every == 0:
                ckpt_path = out_dir / "checkpoints" / f"ep_{ep+1:04d}.pt"
                agent.save_inference_checkpoint(ckpt_path, extra={"episode": ep + 1})

            # Console summary каждые summary_every: rolling-mean returns + action distribution.
            if (ep + 1) % summary_every == 0:
                mean_R = float(np.mean(recent_returns))
                mean_q_inv = float(np.mean(recent_abs_inv))
                print(f"\n  [summary ep={ep+1}] last {len(recent_returns)} eps:")
                print(f"    mean R = {mean_R:+8.2f}   mean |q|̄ = {mean_q_inv:5.1f}")
                print("    action distribution:")
                print(_format_action_hist(recent_action_hist.tolist(), gamma_set))
                recent_action_hist[:] = 0

            # Eval каждые eval_every (после warmup).
            if (ep + 1) % eval_every == 0 and ep >= warmup_episodes:
                eval_metrics = evaluate(agent, lambda_inv=lambda_inv,
                                        build_overrides=build_overrides)
                mean_r = eval_metrics["mean_return"]
                with eval_log_path.open("a") as f:
                    f.write(json.dumps({"ep": ep + 1, **eval_metrics}) + "\n")
                marker = ""
                if mean_r > best_eval:
                    best_eval = mean_r
                    agent.save_inference_checkpoint(
                        out_dir / "best.pt",
                        extra={"episode": ep + 1, "eval_mean_return": mean_r},
                    )
                    marker = " ★ NEW BEST"
                print(f"  [eval ep={ep+1}] mean_R = {mean_r:+8.2f}{marker}")
                for sc, m in eval_metrics["per_scenario"].items():
                    print(f"    {sc:11s}: R={m['return']:+8.2f}  "
                          f"|q|̄={m['mean_abs_inventory']:5.1f}  "
                          f"PnL_eq={m['equity_pnl_cents']/100:+8.0f}c")
                print()
                _save_last(ep + 1)   # snapshot после eval (обновляется best_eval)

    except KeyboardInterrupt:
        # `last.pt` сохраняется после каждого ПОЛНОСТЬЮ завершённого эпизода;
        # Ctrl+C между эпизодами не теряет состояние, Ctrl+C внутри теряет максимум один.
        print(f"\n\n[interrupted] last.pt: эпизод {start_ep}..{ep}")
        print(f"  Resume:  python -m src.rl.train --resume {out_dir}")
        sys.exit(130)

    print(f"\nTraining done. Total wall-clock: {(time.time() - t0)/60:.1f} min")
    print(f"Best eval return: {best_eval:+.2f}")
    print(f"Checkpoints in: {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None,
                   help="Путь к out_dir предыдущего запуска. При задании все остальные "
                        "гиперпараметры игнорируются и берутся из его config.json.")
    p.add_argument("--n-episodes", type=int, default=500)
    p.add_argument("--lambda-inv", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma-disc", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=300)
    p.add_argument("--buffer-capacity", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-steps-per-episode", type=int, default=50)
    p.add_argument("--warmup-episodes", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--summary-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    # GLFT-параметры внутренней стратегии. «Вшиваются» в policy через
    # state-distribution → не менять между train и inference.
    p.add_argument("--rl-A", type=float, default=1.0,
                   help="GLFT intensity scale A. Меньше A → больше η → агрессивнее skew. "
                        "Default 1.0 (literature); A=0.3 — победитель sweep'а в comparison.ipynb.")
    p.add_argument("--rl-sigma", type=float, default=4.4)
    p.add_argument("--rl-k", type=float, default=1.5)
    args = p.parse_args()

    train(
        resume_from=args.resume,
        n_episodes=args.n_episodes,
        lambda_inv=args.lambda_inv,
        hidden=args.hidden,
        lr=args.lr,
        gamma_disc=args.gamma_disc,
        tau=args.tau,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        grad_steps_per_episode=args.grad_steps_per_episode,
        warmup_episodes=args.warmup_episodes,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        summary_every=args.summary_every,
        seed=args.seed,
        out_dir=args.out_dir,
        rl_A=args.rl_A,
        rl_sigma=args.rl_sigma,
        rl_k=args.rl_k,
    )


if __name__ == "__main__":
    main()
