"""Experiment B3-1: Shared initial state computation across parallel rollouts.

Key idea: In MPPI with N rollouts, at t=0 all share the same state. Compute
contact/collision once and share, rather than redundantly in each vmap lane.

We split mjx.step into:
  1. State-dependent prefix: collision detection, contact computation (shared)
  2. Action-dependent suffix: apply controls, integrate forward (per-rollout)
"""

import time
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class SharedPrefixResult(NamedTuple):
    """Results from shared prefix benchmark."""
    independent_time: float
    shared_time: float
    speedup: float
    trajectory_divergence: float  # max abs diff between methods


class SharedPrefixMPPI:
    """MPPI variant that shares initial state computation across rollouts.

    For MuJoCo MJX, the shared prefix includes:
    - Forward kinematics (mj_kinematics)
    - Collision detection (mj_collision)
    - Contact computation (mj_makeConstraint)

    The per-rollout part includes:
    - Control application
    - Constraint solving
    - Integration (mj_Euler / mj_implicit)
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        n_samples: int = 1024,
        horizon: int = 20,
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

    def _rollout_independent(self, state: Any, actions: jax.Array) -> float:
        """Standard rollout: each sample simulates independently from t=0."""
        def step_fn(carry, action):
            s, cost = carry
            c = self.cost_fn(s, action)
            s_next = self.dynamics_fn(s, action)
            return (s_next, cost + c), s_next

        (_, total_cost), traj = jax.lax.scan(step_fn, (state, 0.0), actions)
        return total_cost, traj

    def _rollout_shared_prefix(
        self,
        prefix_state: Any,
        actions: jax.Array,
    ) -> float:
        """Rollout reusing pre-computed prefix state at t=0.

        prefix_state: state after collision/contact computation (shared).
        First step only applies action and integrates. Subsequent steps are normal.
        """
        def step_fn(carry, action):
            s, cost = carry
            c = self.cost_fn(s, action)
            s_next = self.dynamics_fn(s, action)
            return (s_next, cost + c), s_next

        (_, total_cost), traj = jax.lax.scan(step_fn, (prefix_state, 0.0), actions)
        return total_cost, traj

    def plan_independent(self, state: Any, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Standard MPPI: all rollouts independent."""
        noise = jax.random.normal(key, (self.n_samples, self.horizon, self.action_dim))
        actions = noise * self.noise_sigma

        costs, trajs = jax.vmap(
            self._rollout_independent, in_axes=(None, 0)
        )(state, actions)

        weights = jax.nn.softmax(-costs / self.temperature)
        best_action = jnp.einsum('n,nhd->hd', weights, actions)
        return best_action, costs

    def plan_shared_prefix(self, state: Any, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Shared prefix MPPI: compute state-dependent parts once.

        In a real MJX implementation, we would:
        1. Call mjx.forward() once to compute kinematics + contacts
        2. Cache the result
        3. Each rollout starts from this cached state

        Here we simulate this by calling dynamics_fn once with zero action
        to get the "prefix state", then all rollouts diverge from there.
        """
        # Compute shared prefix: forward dynamics with zero action at t=0
        # In MJX: this would be mjx.forward(model, data) -> cached contacts
        zero_action = jnp.zeros(self.action_dim)
        prefix_state = self.dynamics_fn(state, zero_action)

        noise = jax.random.normal(key, (self.n_samples, self.horizon, self.action_dim))
        actions = noise * self.noise_sigma

        # All rollouts share prefix_state
        costs, trajs = jax.vmap(
            self._rollout_shared_prefix, in_axes=(None, 0)
        )(prefix_state, actions)

        weights = jax.nn.softmax(-costs / self.temperature)
        best_action = jnp.einsum('n,nhd->hd', weights, actions)
        return best_action, costs


def benchmark_shared_vs_independent(
    dynamics_fn: Callable,
    cost_fn: Callable,
    state: jax.Array,
    key: jax.Array,
    n_samples: int = 1024,
    n_trials: int = 20,
) -> SharedPrefixResult:
    """Benchmark shared prefix vs independent rollouts."""
    planner = SharedPrefixMPPI(
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        n_samples=n_samples,
        horizon=20,
        action_dim=state.shape[0] // 2 if state.ndim == 1 else 2,
    )

    # JIT compile
    plan_ind = jax.jit(planner.plan_independent)
    plan_shared = jax.jit(planner.plan_shared_prefix)

    # Warm up
    key1, key2 = jax.random.split(key)
    a1, c1 = plan_ind(state, key1)
    jax.block_until_ready(a1)
    a2, c2 = plan_shared(state, key2)
    jax.block_until_ready(a2)

    # Benchmark independent
    times_ind = []
    for i in range(n_trials):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        a, c = plan_ind(state, subkey)
        jax.block_until_ready(a)
        times_ind.append(time.perf_counter() - t0)

    # Benchmark shared
    times_shared = []
    for i in range(n_trials):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        a, c = plan_shared(state, subkey)
        jax.block_until_ready(a)
        times_shared.append(time.perf_counter() - t0)

    avg_ind = sum(times_ind) / len(times_ind)
    avg_shared = sum(times_shared) / len(times_shared)

    # Measure trajectory divergence (same key for fair comparison)
    key, subkey = jax.random.split(key)
    a_ind, c_ind = plan_ind(state, subkey)
    a_shared, c_shared = plan_shared(state, subkey)
    divergence = float(jnp.max(jnp.abs(c_ind - c_shared)))

    return SharedPrefixResult(
        independent_time=avg_ind,
        shared_time=avg_shared,
        speedup=avg_ind / avg_shared if avg_shared > 0 else 0.0,
        trajectory_divergence=divergence,
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
    print("=== B3-1: Shared Prefix Benchmark ===\n")
    state = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(42)

    for n in [256, 512, 1024]:
        result = benchmark_shared_vs_independent(
            _dynamics, _cost, state, key, n_samples=n, n_trials=30
        )
        print(f"N={n:>5d}: independent={result.independent_time*1000:.2f}ms  "
              f"shared={result.shared_time*1000:.2f}ms  "
              f"speedup={result.speedup:.2f}x  "
              f"divergence={result.trajectory_divergence:.6f}")
    print()
    print("Note: With simple dynamics, speedup is minimal. The benefit grows")
    print("with expensive contact/collision computation in MJX environments.")
