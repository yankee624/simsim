"""JAX-native MPPI (Model Predictive Path Integral) implementation.

Fully jit-compilable. Supports warm-starting from previous solution.

Usage:
    mppi = MPPI(dynamics_fn, cost_fn, n_samples=1024, horizon=20, action_dim=2)
    action_seq, info = mppi(state, key)
    # Next step: warm-start
    action_seq, info = mppi(next_state, next_key, prev_mean=action_seq)
"""

from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


class MPPIResult(NamedTuple):
    """Result of MPPI planning."""
    action_sequence: jax.Array   # (horizon, action_dim) - optimal action sequence
    first_action: jax.Array      # (action_dim,) - action to execute now
    costs: jax.Array             # (n_samples,) - costs of all samples
    weights: jax.Array           # (n_samples,) - softmax weights
    best_cost: float             # cost of best sample


class MPPI:
    """Model Predictive Path Integral controller.

    Args:
        dynamics_fn: (state, action) -> next_state
        cost_fn: (state, action) -> scalar cost (to minimize)
        n_samples: number of action sequence samples
        horizon: planning horizon (number of timesteps)
        n_iters: number of MPPI iterations per planning step
        temperature: softmax temperature (lambda). Lower = more greedy.
        noise_sigma: std of action noise
        action_dim: dimensionality of action space
        action_bounds: (low, high) tuple of arrays, optional
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        n_samples: int = 1024,
        horizon: int = 20,
        n_iters: int = 1,
        temperature: float = 1.0,
        noise_sigma: float = 1.0,
        action_dim: int = 2,
        action_bounds: Optional[Tuple[jax.Array, jax.Array]] = None,
    ):
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.n_samples = n_samples
        self.horizon = horizon
        self.n_iters = n_iters
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim
        self.action_bounds = action_bounds

    def _rollout(self, state: Any, actions: jax.Array) -> jax.Array:
        """Rollout a single action sequence and return total cost.

        Args:
            state: initial state
            actions: (horizon, action_dim) action sequence

        Returns:
            total_cost: scalar
        """
        def step_fn(carry, action):
            s, total_cost = carry
            cost = self.cost_fn(s, action)
            s_next = self.dynamics_fn(s, action)
            return (s_next, total_cost + cost), None

        (final_state, total_cost), _ = jax.lax.scan(step_fn, (state, 0.0), actions)
        # Terminal cost
        terminal_cost = self.cost_fn(final_state, jnp.zeros(self.action_dim))
        return total_cost + terminal_cost

    def _evaluate(self, state: Any, all_actions: jax.Array) -> jax.Array:
        """Evaluate all action sequences in parallel.

        Args:
            state: initial state (not batched)
            all_actions: (n_samples, horizon, action_dim)

        Returns:
            costs: (n_samples,)
        """
        return jax.vmap(self._rollout, in_axes=(None, 0))(state, all_actions)

    def _clip_actions(self, actions: jax.Array) -> jax.Array:
        """Clip actions to bounds if specified."""
        if self.action_bounds is not None:
            low, high = self.action_bounds
            actions = jnp.clip(actions, low, high)
        return actions

    def _warm_start(self, prev_mean: Optional[jax.Array]) -> jax.Array:
        """Create initial mean from previous solution (shifted by 1 step)."""
        if prev_mean is None:
            return jnp.zeros((self.horizon, self.action_dim))
        # Shift: drop first action, append zeros at end
        shifted = jnp.concatenate([prev_mean[1:], jnp.zeros((1, self.action_dim))], axis=0)
        return shifted

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        state: Any,
        key: jax.Array,
        prev_mean: Optional[jax.Array] = None,
    ) -> MPPIResult:
        """Run MPPI planning.

        Args:
            state: current state
            key: PRNG key
            prev_mean: previous optimal action sequence for warm-starting

        Returns:
            MPPIResult with optimal action sequence and diagnostics
        """
        mean = self._warm_start(prev_mean)

        def iteration(carry, _):
            mean, key = carry
            key, subkey = jax.random.split(key)

            # Sample action perturbations
            noise = jax.random.normal(subkey, (self.n_samples, self.horizon, self.action_dim))
            noise = noise * self.noise_sigma
            actions = mean[None, :, :] + noise  # (n_samples, horizon, action_dim)
            actions = self._clip_actions(actions)

            # Evaluate all samples
            costs = self._evaluate(state, actions)

            # Compute weights (softmax with temperature)
            # Shift costs for numerical stability
            min_cost = jnp.min(costs)
            weights = jax.nn.softmax(-(costs - min_cost) / self.temperature)

            # Weighted mean of action sequences
            new_mean = jnp.einsum('n,nhd->hd', weights, actions)

            return (new_mean, key), (costs, weights)

        (mean, _), (all_costs, all_weights) = jax.lax.scan(
            iteration, (mean, key), None, length=self.n_iters
        )

        # Use last iteration's costs/weights
        costs = all_costs[-1]
        weights = all_weights[-1]
        mean = self._clip_actions(mean)

        return MPPIResult(
            action_sequence=mean,
            first_action=mean[0],
            costs=costs,
            weights=weights,
            best_cost=jnp.min(costs),
        )


# --- Demo: Double integrator (point mass to target) ---

def _demo_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    """Double integrator: state = [x, y, vx, vy], action = [ax, ay]."""
    dt = 0.1
    pos = state[:2] + state[2:4] * dt + 0.5 * action * dt ** 2
    vel = state[2:4] + action * dt
    return jnp.concatenate([pos, vel])


def _demo_cost(state: jax.Array, action: jax.Array) -> float:
    """Cost: distance to origin + control effort."""
    target = jnp.array([1.0, 1.0, 0.0, 0.0])
    state_cost = jnp.sum((state - target) ** 2)
    action_cost = 0.01 * jnp.sum(action ** 2)
    return state_cost + action_cost


if __name__ == "__main__":
    import time

    print("=== MPPI Demo: Point mass to target ===")

    mppi = MPPI(
        dynamics_fn=_demo_dynamics,
        cost_fn=_demo_cost,
        n_samples=1024,
        horizon=20,
        n_iters=3,
        temperature=0.5,
        noise_sigma=2.0,
        action_dim=2,
        action_bounds=(jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0])),
    )

    state = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(0)

    # Warm up JIT
    result = mppi(state, key)
    jax.block_until_ready(result.first_action)

    # Run MPC loop
    prev_mean = None
    trajectory = [state]

    start = time.perf_counter()
    for t in range(50):
        key, subkey = jax.random.split(key)
        result = mppi(state, subkey, prev_mean=prev_mean)
        jax.block_until_ready(result.first_action)

        state = _demo_dynamics(state, result.first_action)
        prev_mean = result.action_sequence
        trajectory.append(state)

        if t % 10 == 0:
            dist = jnp.linalg.norm(state[:2] - jnp.array([1.0, 1.0]))
            print(f"  t={t:3d}  pos=({state[0]:.3f}, {state[1]:.3f})  "
                  f"dist={dist:.4f}  best_cost={result.best_cost:.4f}")

    elapsed = time.perf_counter() - start
    print(f"\n50 MPC steps in {elapsed:.3f}s ({50/elapsed:.1f} steps/sec)")
    final_dist = jnp.linalg.norm(state[:2] - jnp.array([1.0, 1.0]))
    print(f"Final distance to target: {final_dist:.4f}")
