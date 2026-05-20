"""DQN-агент: MLP Q-сеть (2×hidden ReLU) + target network с soft Polyak
update + Huber loss на TD-target + ε-greedy. `load_policy_for_inference`
возвращает `callable(state) -> int` для использования в симуляции через
флаг `--rl-policy-path`."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class DQNAgent:

    def __init__(
        self,
        state_dim,
        n_actions,
        gamma_set,
        hidden=64,
        gamma_disc=0.99,
        lr=3e-4,
        tau=0.005,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=10_000,
        device="cpu",
        seed=0,
    ):
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma_set = list(gamma_set)
        self.hidden = hidden
        self.gamma_disc = gamma_disc
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.device = torch.device(device)
        self.rng = np.random.RandomState(seed)

        self.policy_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.step_count = 0  # для eps decay schedule

    @property
    def epsilon(self):
        progress = min(1.0, self.step_count / max(1, self.eps_decay_steps))
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    def act_greedy(self, state):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            return int(self.policy_net(x).argmax(dim=-1).item())

    def act_eps_greedy(self, state):
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(0, self.n_actions))
        return self.act_greedy(state)

    def make_eps_greedy_policy(self):
        return lambda s: self.act_eps_greedy(s)

    def make_greedy_policy(self):
        return lambda s: self.act_greedy(s)

    def train_step(self, batch):
        s = torch.from_numpy(batch["states"]).to(self.device)
        a = torch.from_numpy(batch["actions"]).to(self.device)
        r = torch.from_numpy(batch["rewards"]).to(self.device)
        s_next = torch.from_numpy(batch["next_states"]).to(self.device)
        d = torch.from_numpy(batch["dones"]).to(self.device)

        q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma_disc * self.target_net(s_next).max(dim=1).values * (1.0 - d)

        loss = F.smooth_l1_loss(q_pred, target)   # Huber: стабильнее MSE на outliers
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft Polyak update: target ← τ·policy + (1-τ)·target
        with torch.no_grad():
            for p_t, p_p in zip(self.target_net.parameters(),
                                self.policy_net.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_p.data)

        self.step_count += 1
        return {
            "loss": float(loss.item()),
            "mean_q": float(q_pred.mean().item()),
            "epsilon": self.epsilon,
        }

    # -------- Checkpoint I/O --------

    def save_inference_checkpoint(self, path, extra=None):
        """Минимальный ckpt: policy weights + архитектурные мета.
        Используется для `best.pt` и `ep_NNNN.pt`."""
        payload = {
            "state_dict": self.policy_net.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "gamma_set": list(self.gamma_set),
            "gamma_disc": self.gamma_disc,
            "step_count": self.step_count,
        }
        if extra:
            payload["extra"] = extra
        _atomic_torch_save(payload, path)

    def save_training_state(self, path, **extra):
        """Полный snapshot для resume: policy + target + optimizer + RNG +
        step_count + `extra` (episode, best_eval, buffer_state, ...).
        Атомарная запись через `.tmp` + `os.replace`."""
        payload = {
            "state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": self.rng.get_state(),
            "step_count": self.step_count,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "gamma_set": list(self.gamma_set),
            "gamma_disc": self.gamma_disc,
            **extra,
        }
        _atomic_torch_save(payload, path)

    def load_training_state(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["state_dict"])
        # target_state_dict отсутствует в inference-ckpt'ах — fallback на policy.
        self.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["state_dict"]))
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "rng_state" in ckpt:
            self.rng.set_state(ckpt["rng_state"])
        self.step_count = ckpt.get("step_count", 0)
        return ckpt


def _atomic_torch_save(payload, path):
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_policy_for_inference(path, device="cpu"):
    """Возвращает `callable(state: np.ndarray) -> int` (argmax Q, без
    exploration). Используется через `--rl-policy-path` в сценарии."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = QNetwork(
        state_dim=ckpt["state_dim"],
        n_actions=ckpt["n_actions"],
        hidden=ckpt.get("hidden", 64),
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    @torch.no_grad()
    def policy_fn(state):
        x = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)
        q = net(x)
        return int(q.argmax(dim=-1).item())

    return policy_fn
