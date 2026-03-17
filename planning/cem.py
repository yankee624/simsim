"""JAX-native CEM (Cross-Entropy Method) implementation.

Fully jit-compilable. Supports warm-starting from previous solution.

Usage:
    cem = CEM(dynamics_fn, cost_fn, n_samples=512, horizon=20, action_dim=2)
    action_seq, info = cem(state, key)
"""

from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


class CEMResult(NamedTuple):
    """Result of CEM planning."""
    action_sequence: jax.Array   # (horizon, action_dim) - best action sequence
    first_action: jax.Array      # (action_dim,) - action to execute now
    mean: jax.Array              # (horizon, action_dim) - final distribution mean
    std: jax.Array               # (horizon, action_dim) - final distribution std
    best_cost: float


class CEM:
    """Cross-Entropy Method planner.

    Args:
        dynamics_fn: (state, action) -> next_state
        cost_fn: (state, action) -> scalar cost (to minimize)
        n_samples: number of samples per iteration
        horizon: planning horizon
        n_elites: number of elite samples to keep
        n_iters: number of CEM iterations
        action_dim: dimensionality of action space
        action_bounds: (low, high) tuple, optional
        init_std: initial standard deviation
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        n_samples: int = 512,
        horizon: int = 20,
        n_elites: int = 64,
        n_iters: int = 5,
        action_dim: int = 2,
        action_bounds: Optional[Tuple[jax.Array, jax.Array]] = None,
        init_std: float = 1.0,
    ):
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.n_samples = n_samples
        self.horizon = horizon
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.init_std = init_std

    def _rollout(self, state: Any, actions: jax.Array) -> jax.Array:
        """Rollout a single trajectory, return total cost."""
        def step_fn(carry, action):
            s, total_cost = carry
            cost = self.cost_fn(s, action)
            s_next = self.dynamics_fn(s, action)
            return (s_next, total_cost + cost), None

        (final_state, total_cost), _ = jax.lax.scan(step_fn, (state, 0.0), actions)
        terminal_cost = self.cost_fn(final_state, jnp.zeros(self.action_dim))
        return total_cost + terminal_cost

    def _evaluate(self, state: Any, all_actions: jax.Array) -> jax.Array:
        """Batched evaluation using vmap."""
        return jax.vmap(self._rollout, in_axes=(None, 0))(state, all_actions)

    def _clip_actions(self, actions: jax.Array) -> jax.Array:
        if self.action_bounds is not None:
            low, high = self.action_bounds
            actions = jnp.clip(actions, low, high)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        state: Any,
        key: jax.Array,
        prev_mean: Optional[jax.Array] = None,
        prev_std: Optional[jax.Array] = None,
    ) -> CEMResult:
        """Run CEM planning.

        Args:
            state: current state
            key: PRNG key
            prev_mean: warm-start mean (shifted from previous step)
            prev_std: warm-start std

        Returns:
            CEMResult with best action sequence and distribution params
        """
        # Initialize distribution
        if prev_mean is not None:
            # Warm start: shift previous solution
            mean = jnp.concatenate([prev_mean[1:],
                                    jnp.zeros((1, self.action_dim))], axis=0)
        else:
            mean = jnp.zeros((self.horizon, self.action_dim))

        if prev_std is not None:
            std = prev_std
        else:
            std = jnp.full((self.horizon, self.action_dim), self.init_std)

        def iteration(carry, _):
            mean, std, key = carry
            key, subkey = jax.random.split(key)

            # Sample from current distribution
            noise = jax.random.normal(subkey, (self.n_samples, self.horizon, self.action_dim))
            actions = mean[None, :, :] + std[None, :, :] * noise
            actions = self._clip_actions(actions)

            # Evaluate
            costs = self._evaluate(state, actions)

            # Select elites (lowest cost)
            elite_indices = jnp.argsort(costs)[:self.n_elites]
            elite_actions = actions[elite_indices]

            # Refit distribution to elites
            new_mean = jnp.mean(elite_actions, axis=0)
            new_std = jnp.std(elite_actions, axis=0) + 1e-6  # prevent collapse

            best_cost = costs[elite_indices[0]]

            return (new_mean, new_std, key), best_cost

        (mean, std, _), all_best_costs = jax.lax.scan(
            iteration, (mean, std, key), None, length=self.n_iters
        )

        mean = self._clip_actions(mean)

        return CEMResult(
            action_sequence=mean,
            first_action=mean[0],
            mean=mean,
            std=std,
            best_cost=all_best_costs[-1],
        )


# --- Demo: Same double integrator as MPPI ---

def _demo_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    dt = 0.1
    pos = state[:2] + state[2:4] * dt + 0.5 * action * dt ** 2
    vel = state[2:4] + action * dt
    return jnp.concatenate([pos, vel])


def _demo_cost(state: jax.Array, action: jax.Array) -> float:
    target = jnp.array([1.0, 1.0, 0.0, 0.0])
    return jnp.sum((state - target) ** 2) + 0.01 * jnp.sum(action ** 2)


if __name__ == "__main__":
    import time

    print("=== CEM Demo: Point mass to target ===")

    cem = CEM(
        dynamics_fn=_demo_dynamics,
        cost_fn=_demo_cost,
        n_samples=512,
        horizon=20,
        n_elites=64,
        n_iters=5,
        action_dim=2,
        action_bounds=(jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0])),
    )

    state = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(0)

    # Warm up
    result = cem(state, key)
    jax.block_until_ready(result.first_action)

    prev_mean, prev_std = None, None
    start = time.perf_counter()
    for t in range(50):
        key, subkey = jax.random.split(key)
        result = cem(state, subkey, prev_mean=prev_mean, prev_std=prev_std)
        jax.block_until_ready(result.first_action)

        state = _demo_dynamics(state, result.first_action)
        prev_mean = result.action_sequence
        prev_std = result.std

        if t % 10 == 0:
            dist = jnp.linalg.norm(state[:2] - jnp.array([1.0, 1.0]))
            print(f"  t={t:3d}  pos=({state[0]:.3f}, {state[1]:.3f})  "
                  f"dist={dist:.4f}  best_cost={result.best_cost:.4f}")

    elapsed = time.perf_counter() - start
    print(f"\n50 MPC steps in {elapsed:.3f}s ({50/elapsed:.1f} steps/sec)")
    final_dist = jnp.linalg.norm(state[:2] - jnp.array([1.0, 1.0]))
    print(f"Final distance to target: {final_dist:.4f}")
