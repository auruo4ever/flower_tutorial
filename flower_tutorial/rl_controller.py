# rl_agent.py
"""
Improved RLController — richer state + stable multi-step training.

API:
    agent = RLController(num_metrics=3, max_clients=100)
    k, thresholds = agent.decide(quality_matrix, min_clients=3, max_clients=20)
    reward = agent.compute_reward(acc_t, selected_ids)
    diag = agent.update()   # call every round; internal buffer updates when enough samples collected
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

class RLController:
    def __init__(
        self,
        num_metrics: int,
        max_clients: int,
        lr: float = 3e-4,
        beta1: float = 1.0,        # accuracy weight
        beta2: float = 0.3,        # fairness weight
        beta3: float = 0.05,       # comm cost weight
        gamma: float = 0.99,       # discount for multi-step returns
        update_freq: int = 1,      # accumulate this many rounds before updating
        hidden: int = 128,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device("cpu")
        self.num_metrics = num_metrics
        self.max_clients = int(max_clients)

        # --- state size:
        # per-metric: min, mean, max, std  => 4 * num_metrics
        # temporal extras: prev_acc, prev_loss, prev_reward => 3
        # prev_k (scalar), prev_thresholds (num_metrics), round_index (scalar)
        self.state_size = 4 * num_metrics + 3 + 1 + num_metrics + 1

        # actor outputs: 1 (proportion p) + num_metrics (thresholds)
        self.action_size = 1 + num_metrics

        # networks
        self.actor = nn.Sequential(
            nn.Linear(self.state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.action_size),
            nn.Sigmoid()  # outputs in (0,1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(self.state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ).to(self.device)

        # learnable log std (scalar) for action sampling in pre-sigmoid space
        self.log_std = nn.Parameter(torch.tensor(0.0, device=self.device))

        # optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std],
            lr=lr
        )

        # hyperparams
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.beta3 = float(beta3)
        self.gamma = float(gamma)
        self.update_freq = int(update_freq)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

        # bookkeeping for temporal state
        self.prev_acc = None
        self.prev_loss = None
        self.prev_reward = 0.0
        self.prev_thresholds = np.zeros(self.num_metrics, dtype=np.float32)
        self.prev_k = 0.0
        self.round_idx = 0

        # fairness mapping for arbitrary client IDs
        self.wait = {}  # dict: client_id -> rounds since last selected

        # smoothed accuracy for low-noise reward
        self.acc_ema = None
        self.ema_alpha = 0.2

        # buffers for multi-step updates
        self._buf_logprob = []
        self._buf_value = []
        self._buf_entropy = []
        self._buf_reward = []

        # last action (for storing after decide)
        self._last_action = None  # torch tensor action_squashed
        self._last_logprob = None  # tensor
        self._last_value = None    # tensor

    # --------------------------
    # State construction
    # --------------------------
    def _state_from_quality(self, q: np.ndarray) -> torch.Tensor:
        """
        q: np.ndarray shape (num_clients, num_metrics)
        returns: torch tensor shape (1, state_size)
        """
        q = np.asarray(q, dtype=np.float32)
        # safety for degenerate shapes
        if q.ndim == 1:
            q = q.reshape(1, -1)

        mins  = np.min(q, axis=0)
        means = np.mean(q, axis=0)
        maxs  = np.max(q, axis=0)
        stds  = np.std(q, axis=0) + 1e-8

        # temporal scalars
        prev_acc = 0.0 if self.prev_acc is None else float(self.prev_acc)
        prev_loss = 0.0 if self.prev_loss is None else float(self.prev_loss)
        prev_reward = float(self.prev_reward)

        # previous thresholds and k
        prev_thr = self.prev_thresholds.astype(np.float32)
        prev_k = np.array([float(self.prev_k)], dtype=np.float32)

        # round index scaled
        round_scaled = np.array([self.round_idx / 1000.0], dtype=np.float32)

        state = np.concatenate([
            mins, means, maxs, stds,
            np.array([prev_acc, prev_loss, prev_reward], dtype=np.float32),
            prev_thr,
            prev_k,
            round_scaled
        ], axis=0).astype(np.float32)

        # make sure it's the expected size
        assert state.shape[0] == self.state_size, f"state size mismatch {state.shape[0]} vs {self.state_size}"

        return torch.from_numpy(state).unsqueeze(0).to(self.device)

    # --------------------------
    # Decision (actor forward + sampling)
    # --------------------------
    def decide(
        self,
        quality_matrix: np.ndarray,
        min_clients: int,
        max_clients: Optional[int] = None,
        deterministic: bool = False,
    ) -> Tuple[int, np.ndarray]:
        """
        Returns (k, thresholds)
        Also stores tensors needed for later update (logprob, value, entropy).
        """
        self.round_idx += 1
        if max_clients is None:
            max_clients = int(quality_matrix.shape[0])

        state = self._state_from_quality(quality_matrix)  # (1, state_size)

        # forward through actor & critic 
        means = self.actor(state).squeeze(0)   # (action_size,)
        value = self.critic(state).squeeze(0)  # scalar tensor

        # sampling in pre-sigmoid (logit) space for proper logprob
        if deterministic:
            action_squashed = means
            logprob_tensor = None
            entropy_tensor = None
        else:
            eps = torch.randn_like(means, device=self.device)
            # invert sigmoid to get mu in R
            mu = torch.logit(means.clamp(1e-6, 1 - 1e-6))
            std = torch.exp(self.log_std)
            x = mu + std * eps   # sample in R
            action_squashed = torch.sigmoid(x)  # in (0,1)

            # logprob of x under Normal(mu, std)
            var = std * std
            logp_comp = -0.5 * (((x - mu) ** 2) / (var + 1e-8) + 2.0 * self.log_std + math.log(2 * math.pi))
            logprob_tensor = torch.sum(logp_comp)  # scalar tensor
            # entropy for normal dist
            entropy_tensor = torch.sum(0.5 * (torch.log(2 * math.pi * var) + 1.0))

        # interpret action
        p = float(action_squashed[0].clamp(0.0, 1.0).item())
        thresholds = action_squashed[1 : 1 + self.num_metrics].detach().cpu().numpy()

        raw_k = int(round(p * float(max_clients)))
        k = max(min_clients, min(raw_k, max_clients))

        # store tensors for later
        self._last_action = action_squashed.detach().cpu().numpy()
        self._last_logprob = logprob_tensor
        self._last_value = value
        self._last_entropy = entropy_tensor

        # Also store prev decisions for next state's temporal features
        self.prev_thresholds = thresholds.copy()
        self.prev_k = float(k) / float(self.max_clients)

        return int(k), thresholds

    # --------------------------
    # Reward calculation (smoothing + fairness + comm cost + variance penalty)
    # --------------------------
    def compute_reward(self, acc_t: float, selected_ids) -> float:
        """
        Compute shaped reward using:
          - smoothed accuracy improvement (EMA)
          - fairness penalty: avg wait
          - communication benefit: favor smaller k (beta3)
          - optional variance penalty (small)
        selected_ids: iterable of client IDs (could be ints)
        """
        # update EMA
        if self.acc_ema is None:
            self.acc_ema = float(acc_t)
        else:
            self.acc_ema = (1.0 - self.ema_alpha) * self.acc_ema + self.ema_alpha * float(acc_t)

        # accuracy delta on smoothed signal
        if self.prev_acc is None:
            delta_acc = 0.0
        else:
            delta_acc = float(self.acc_ema - self.prev_acc)

        # update prev_acc for next time (we keep raw acc too)
        self.prev_acc = float(self.acc_ema)

        # fairness: maintain wait counters (dict) for arbitrary IDs
        for cid in list(self.wait.keys()):
            self.wait[cid] += 1
        for cid in selected_ids:
            if cid not in self.wait:
                self.wait[cid] = 0
            self.wait[cid] = 0

        if len(self.wait) == 0:
            avg_wait = 0.0
            std_wait = 0.0
        else:
            waits = np.fromiter(self.wait.values(), dtype=float)
            avg_wait = float(waits.mean())
            std_wait = float(waits.std())

        # communication cost: smaller k gets positive reward when normalized by max_clients
        # we derive k from prev_k (stored normalized)
        comm_reward = (1.0 - float(self.prev_k))  # higher when prev_k small

        # small penalty for large variance (to favor stable selection)
        var_penalty = -0.001 * std_wait

        # combine
        r_acc = self.beta1 * delta_acc
        r_fair = - self.beta2 * avg_wait
        r_comm = self.beta3 * comm_reward

        reward = float(r_acc + r_fair + r_comm + var_penalty)

        # store for temporal features
        self.prev_reward = reward

        # buffer reward for update
        self._buf_reward.append(reward)

        return reward

    # --------------------------
    # Update / learning
    # --------------------------
    def update(self) -> dict:
        """
        Append last logprob/value/entropy (from decide) and use buffered rewards to update
        when buffer length >= update_freq. Returns diagnostics dict (possibly empty).
        Usage: call update() each round after compute_reward(); it will only perform
        gradient step every self.update_freq rounds.
        """

        # If no last values (e.g., decide might have been deterministic), still append zeros to keep timing
        if self._last_logprob is None or self._last_value is None:
            # append placeholders so timings align
            self._buf_logprob.append(torch.tensor(0.0, device=self.device))
            self._buf_value.append(torch.tensor(0.0, device=self.device))
            self._buf_entropy.append(torch.tensor(0.0, device=self.device))
        else:
            # store tensors (keep computation graph)
            self._buf_logprob.append(self._last_logprob)
            self._buf_value.append(self._last_value)
            # entropy may be None (deterministic); use zero
            self._buf_entropy.append(self._last_entropy if self._last_entropy is not None else torch.tensor(0.0, device=self.device))

        # clear last placeholders
        self._last_logprob = None
        self._last_value = None
        self._last_entropy = None
        self._last_action = None

        # if not enough samples, skip update
        if len(self._buf_reward) < self.update_freq:
            return {}

        # Prepare tensors
        rewards = self._buf_reward[:]  # python list of floats
        logprobs = self._buf_logprob[:]  # list of tensors (possibly requires_grad)
        values = self._buf_value[:]  # list of tensors
        entropies = self._buf_entropy[:]  # list of tensors

        # compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        values_tensor = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for v in values]).squeeze()
        logprob_tensor = torch.stack([lp if isinstance(lp, torch.Tensor) else torch.tensor(lp, device=self.device) for lp in logprobs])
        entropy_tensor = torch.stack([e if isinstance(e, torch.Tensor) else torch.tensor(e, device=self.device) for e in entropies])

        # advantage
        advantages = returns - values_tensor.detach()

        # actor loss: - E[ logpi * A ] - entropy_coef * entropy
        actor_loss = - (logprob_tensor * advantages).mean() - self.entropy_coef * entropy_tensor.mean()

        # critic loss: MSE
        critic_loss = self.value_coef * (advantages.pow(2).mean())

        loss = actor_loss + critic_loss

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], max_norm=self.max_grad_norm)
        self.optimizer.step()

        # diagnostics
        diag = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "total_loss": float(loss.item()),
            "mean_return": float(returns.mean().item()),
            "last_reward": float(rewards[-1]),
        }

        # clear buffers
        self._buf_logprob.clear()
        self._buf_value.clear()
        self._buf_entropy.clear()
        self._buf_reward.clear()

        return diag

    # --------------------------
    # Save / load helpers (optional)
    # --------------------------
    def save(self, path: str):
        data = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": float(self.log_std.detach().cpu().numpy()),
            "prev_thresholds": self.prev_thresholds,
            "prev_k": self.prev_k,
            "round_idx": self.round_idx
        }
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        with torch.no_grad():
            self.log_std.copy_(torch.tensor(data.get("log_std", 0.0), device=self.device))
        self.prev_thresholds = data.get("prev_thresholds", self.prev_thresholds)
        self.prev_k = data.get("prev_k", self.prev_k)
        self.round_idx = data.get("round_idx", self.round_idx)
