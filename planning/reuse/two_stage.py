"""Experiment B3-4: Two-stage coarse-to-fine evaluation pipeline.

Round 1: 1024 candidates at short horizon (H=5) -> select top 64
Round 2: 64 candidates at long horizon (H=30) -> final selection

Hypothesis: Same compute budget yields better decisions than single-stage.
"""

import time
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


class TwoStageResult(NamedTuple):
    """Result of two-stage planning."""
    action_sequence: jax.Array
    first_action: jax.Array
    round1_costs: jax.Array
    round2_costs: jax.Array
    selected_indices: jax.Array


class BenchmarkResult(NamedTuple):
    """Benchmark comparison between single-stage and two-stage."""
    single_time: float
    two_stage_time: float
    single_best_cost: float
    two_stage_best_cost: float
    single_total_reward: float
    two_stage_total_reward: float


class TwoStageMPPI:
    """MPPI with two-stage coarse-to-fine evaluation.

    Stage 1 (coarse): Evaluate many candidates at short horizon
    Stage 2 (fine): Re-evaluate top candidates at full horizon
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        n_samples_r1: int = 1024,
        n_samples_r2: int = 64,
        horizon_r1: int = 5,
        horizon_r2: int = 30,
        temperature: float = 1.0,
        noise_sigma: float = 1.0,
        action_dim: int = 2,
        action_bounds: Optional[Tuple[jax.Array, jax.Array]] = None,
    ):
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.n_samples_r1 = n_samples_r1
        self.n_samples_r2 = n_samples_r2
        self.horizon_r1 = horizon_r1
        self.horizon_r2 = horizon_r2
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim
        self.action_bounds = action_bounds

    def _rollout(self, state: Any, actions: jax.Array) -> float:
        """Single trajectory rollout for given horizon."""
        def step_fn(carry, action):
            s, cost = carry
            c = self.cost_fn(s, action)
            s_next = self.dynamics_fn(s, action)
            return (s_next, cost + c), None

        (final_state, total_cost), _ = jax.lax.scan(step_fn, (state, 0.0), actions)
        terminal_cost = self.cost_fn(final_state, jnp.zeros(self.action_dim))
        return total_cost + terminal_cost

    def _clip(self, actions):
        if self.action_bounds is not None:
            actions = jnp.clip(actions, self.action_bounds[0], self.action_bounds[1])
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        state: Any,
        key: jax.Array,
        prev_mean: Optional[jax.Array] = None,
    ) -> TwoStageResult:
        """Run two-stage MPPI planning."""
        # Initialize mean
        if prev_mean is not None:
            mean = jnp.concatenate([prev_mean[1:],
                                    jnp.zeros((1, self.action_dim))], axis=0)
            # Pad or truncate to horizon_r2
            if mean.shape[0] < self.horizon_r2:
                pad = jnp.zeros((self.horizon_r2 - mean.shape[0], self.action_dim))
                mean = jnp.concatenate([mean, pad], axis=0)
            else:
                mean = mean[:self.horizon_r2]
        else:
            mean = jnp.zeros((self.horizon_r2, self.action_dim))

        key1, key2 = jax.random.split(key)

        # === Round 1: Coarse evaluation ===
        # Sample full-horizon actions but evaluate only first horizon_r1 steps
        noise_r1 = jax.random.normal(
            key1, (self.n_samples_r1, self.horizon_r2, self.action_dim)
        )
        actions_full = self._clip(mean[None] + noise_r1 * self.noise_sigma)

        # Evaluate at short horizon
        actions_r1 = actions_full[:, :self.horizon_r1, :]
        costs_r1 = jax.vmap(self._rollout, in_axes=(None, 0))(state, actions_r1)

        # === Filter: select top n_samples_r2 ===
        top_indices = jnp.argsort(costs_r1)[:self.n_samples_r2]
        selected_actions = actions_full[top_indices]  # (n_r2, horizon_r2, action_dim)

        # === Round 2: Fine evaluation ===
        costs_r2 = jax.vmap(self._rollout, in_axes=(None, 0))(state, selected_actions)

        # MPPI weighting on round 2 results
        weights = jax.nn.softmax(-costs_r2 / self.temperature)
        best_action_seq = jnp.einsum('n,nhd->hd', weights, selected_actions)
        best_action_seq = self._clip(best_action_seq)

        return TwoStageResult(
            action_sequence=best_action_seq,
            first_action=best_action_seq[0],
            round1_costs=costs_r1,
            round2_costs=costs_r2,
            selected_indices=top_indices,
        )


class SingleStageMPPI:
    """Standard single-stage MPPI for comparison."""

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        n_samples: int = 1024,
        horizon: int = 30,
        temperature: float = 1.0,
        noise_sigma: float = 1.0,
        action_dim: int = 2,
    ):
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.n_samples = n_samples
        self.horizon = horizon
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim

    def _rollout(self, state, actions):
        def step_fn(carry, action):
            s, cost = carry
            c = self.cost_fn(s, action)
            s_next = self.dynamics_fn(s, action)
            return (s_next, cost + c), None
        (final_state, total_cost), _ = jax.lax.scan(step_fn, (state, 0.0), actions)
        terminal_cost = self.cost_fn(final_state, jnp.zeros(self.action_dim))
        return total_cost + terminal_cost

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, state, key, prev_mean=None):
        if prev_mean is not None:
            mean = jnp.concatenate([prev_mean[1:],
                                    jnp.zeros((1, self.action_dim))], axis=0)
        else:
            mean = jnp.zeros((self.horizon, self.action_dim))

        noise = jax.random.normal(key, (self.n_samples, self.horizon, self.action_dim))
        actions = mean[None] + noise * self.noise_sigma
        costs = jax.vmap(self._rollout, in_axes=(None, 0))(state, actions)
        weights = jax.nn.softmax(-costs / self.temperature)
        best_action = jnp.einsum('n,nhd->hd', weights, actions)
        return best_action, costs


def benchmark_two_stage(
    dynamics_fn: Callable,
    cost_fn: Callable,
    initial_state: jax.Array,
    key: jax.Array,
    n_mpc_steps: int = 50,
) -> BenchmarkResult:
    """Compare single-stage vs two-stage under matched compute budget.

    Single-stage: 1024 samples * H=30 = 30720 rollout steps
    Two-stage: 1024 * H=5 + 64 * H=30 = 5120 + 1920 = 7040 rollout steps
    (Two-stage uses ~4.4x fewer total steps -> test if quality is maintained)
    """
    action_dim = initial_state.shape[0] // 2

    two_stage = TwoStageMPPI(
        dynamics_fn=dynamics_fn, cost_fn=cost_fn,
        n_samples_r1=1024, n_samples_r2=64,
        horizon_r1=5, horizon_r2=30,
        noise_sigma=2.0, action_dim=action_dim,
    )
    single_stage = SingleStageMPPI(
        dynamics_fn=dynamics_fn, cost_fn=cost_fn,
        n_samples=1024, horizon=30,
        noise_sigma=2.0, action_dim=action_dim,
    )

    # Warm up JIT
    key, k1, k2 = jax.random.split(key, 3)
    r1 = two_stage(initial_state, k1)
    jax.block_until_ready(r1.first_action)
    r2 = single_stage(initial_state, k2)
    jax.block_until_ready(r2[0])

    # --- Single stage MPC ---
    state = initial_state
    prev_mean = None
    total_reward_single = 0.0
    t0 = time.perf_counter()
    for _ in range(n_mpc_steps):
        key, k1 = jax.random.split(key)
        action_seq, costs = single_stage(state, k1, prev_mean)
        jax.block_until_ready(action_seq)
        action = action_seq[0]
        total_reward_single -= float(cost_fn(state, action))
        state = dynamics_fn(state, action)
        prev_mean = action_seq
    single_time = (time.perf_counter() - t0) / n_mpc_steps
    single_best_cost = float(jnp.min(costs))

    # --- Two stage MPC ---
    state = initial_state
    prev_mean = None
    total_reward_two = 0.0
    t0 = time.perf_counter()
    for _ in range(n_mpc_steps):
        key, k1 = jax.random.split(key)
        result = two_stage(state, k1, prev_mean)
        jax.block_until_ready(result.first_action)
        action = result.first_action
        total_reward_two -= float(cost_fn(state, action))
        state = dynamics_fn(state, action)
        prev_mean = result.action_sequence
    two_time = (time.perf_counter() - t0) / n_mpc_steps
    two_best_cost = float(jnp.min(result.round2_costs))

    return BenchmarkResult(
        single_time=single_time,
        two_stage_time=two_time,
        single_best_cost=single_best_cost,
        two_stage_best_cost=two_best_cost,
        single_total_reward=total_reward_single,
        two_stage_total_reward=total_reward_two,
    )


# --- Demo ---

def _dynamics(state, action):
    dt = 0.1
    pos = state[:2] + state[2:4] * dt + 0.5 * action * dt**2
    vel = state[2:4] + action * dt
    return jnp.concatenate([pos, vel])

def _cost(state, action):
    target = jnp.array([1.0, 1.0, 0.0, 0.0])
    return jnp.sum((state - target)**2) + 0.01 * jnp.sum(action**2)


if __name__ == "__main__":
    print("=== B3-4: Two-Stage Evaluation Benchmark ===\n")

    state = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(42)

    result = benchmark_two_stage(
        _dynamics, _cost, state, key, n_mpc_steps=50
    )

    print(f"Single-stage (1024 x H=30):")
    print(f"  Time: {result.single_time*1000:.2f} ms/step")
    print(f"  Total reward: {result.single_total_reward:.4f}")
    print(f"  Best cost (last step): {result.single_best_cost:.4f}")
    print()
    print(f"Two-stage (1024xH=5 -> 64xH=30):")
    print(f"  Time: {result.two_stage_time*1000:.2f} ms/step")
    print(f"  Total reward: {result.two_stage_total_reward:.4f}")
    print(f"  Best cost (last step): {result.two_stage_best_cost:.4f}")
    print()
    speedup = result.single_time / result.two_stage_time
    print(f"Speedup: {speedup:.2f}x")
    print(f"Reward ratio: {result.two_stage_total_reward/result.single_total_reward:.4f}")
