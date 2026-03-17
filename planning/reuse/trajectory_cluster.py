"""Experiment B3-2: Cluster similar action sequences, simulate representatives only.

1024 action candidates -> K clusters -> simulate K representatives ->
approximate others via linear extrapolation from nearest representative.

Key metric: rank correlation between approximate and true costs.
"""

import time
from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class ClusterResult(NamedTuple):
    """Results for a single K value."""
    k: int
    full_sim_time: float
    cluster_time: float
    speedup: float
    rank_correlation: float   # Spearman correlation
    cost_mse: float          # MSE between true and approximate costs


class ClusteredMPPI:
    """MPPI with trajectory clustering for reduced simulation cost.

    Instead of simulating all N candidates:
    1. Cluster N actions into K groups
    2. Simulate K representative actions (cluster centroids)
    3. Approximate costs for non-representatives via linearization
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        n_samples: int = 1024,
        n_clusters: int = 32,
        horizon: int = 20,
        temperature: float = 1.0,
        noise_sigma: float = 1.0,
        action_dim: int = 2,
        kmeans_iters: int = 10,
    ):
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.horizon = horizon
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim
        self.kmeans_iters = kmeans_iters

    def _rollout(self, state: Any, actions: jax.Array) -> float:
        """Single trajectory rollout."""
        def step_fn(carry, action):
            s, cost = carry
            c = self.cost_fn(s, action)
            s_next = self.dynamics_fn(s, action)
            return (s_next, cost + c), None

        (_, total_cost), _ = jax.lax.scan(step_fn, (state, 0.0), actions)
        return total_cost

    def cluster_actions(
        self, actions: jax.Array, key: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """K-means clustering of action sequences.

        Args:
            actions: (N, horizon, action_dim)
            key: PRNG key

        Returns:
            centroids: (K, horizon, action_dim)
            assignments: (N,) cluster assignment for each sample
            distances: (N,) distance to assigned centroid
        """
        n = actions.shape[0]
        k = self.n_clusters
        flat_actions = actions.reshape(n, -1)  # (N, horizon*action_dim)

        # Initialize centroids by random selection
        indices = jax.random.choice(key, n, (k,), replace=False)
        centroids = flat_actions[indices]  # (K, horizon*action_dim)

        def kmeans_step(centroids, _):
            # Assign each point to nearest centroid
            # dists: (N, K)
            dists = jnp.sum(
                (flat_actions[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
            )
            assignments = jnp.argmin(dists, axis=1)  # (N,)

            # Recompute centroids
            def update_centroid(c_idx):
                mask = (assignments == c_idx).astype(jnp.float32)
                count = jnp.sum(mask) + 1e-8
                return jnp.sum(flat_actions * mask[:, None], axis=0) / count

            new_centroids = jax.vmap(update_centroid)(jnp.arange(k))
            return new_centroids, assignments

        centroids, assignments = jax.lax.scan(
            lambda c, _: kmeans_step(c, _), centroids, None, length=self.kmeans_iters
        )
        # Final assignment
        dists = jnp.sum(
            (flat_actions[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
        )
        assignments = jnp.argmin(dists, axis=1)
        min_dists = jnp.min(dists, axis=1)

        centroids = centroids.reshape(k, self.horizon, self.action_dim)
        return centroids, assignments, min_dists

    def approximate_costs(
        self,
        representative_costs: jax.Array,
        centroids: jax.Array,
        all_actions: jax.Array,
        assignments: jax.Array,
    ) -> jax.Array:
        """Approximate costs using representative costs + distance-based correction.

        Simple approach: cost ≈ representative_cost + alpha * distance_to_centroid
        (alpha estimated from representative cost variance)
        """
        # Each sample gets its representative's cost
        approx = representative_costs[assignments]

        # Optional: add distance-based correction
        flat_actions = all_actions.reshape(all_actions.shape[0], -1)
        flat_centroids = centroids.reshape(centroids.shape[0], -1)
        distances = jnp.sqrt(jnp.sum(
            (flat_actions - flat_centroids[assignments]) ** 2, axis=-1
        ))

        # Scale factor: estimated from cost variance per unit distance
        cost_std = jnp.std(representative_costs) + 1e-8
        centroid_spread = jnp.mean(distances) + 1e-8
        alpha = cost_std / centroid_spread * 0.1  # conservative correction

        return approx + alpha * distances

    def plan(self, state: Any, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Plan using clustered evaluation."""
        key1, key2 = jax.random.split(key)

        # Sample actions
        noise = jax.random.normal(key1, (self.n_samples, self.horizon, self.action_dim))
        actions = noise * self.noise_sigma

        # Cluster
        centroids, assignments, _ = self.cluster_actions(actions, key2)

        # Simulate representatives only
        rep_costs = jax.vmap(self._rollout, in_axes=(None, 0))(state, centroids)

        # Approximate all costs
        approx_costs = self.approximate_costs(rep_costs, centroids, actions, assignments)

        # MPPI weighting
        weights = jax.nn.softmax(-approx_costs / self.temperature)
        best_action = jnp.einsum('n,nhd->hd', weights, actions)

        return best_action, approx_costs

    def plan_full(self, state: Any, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Standard full evaluation (baseline)."""
        noise = jax.random.normal(key, (self.n_samples, self.horizon, self.action_dim))
        actions = noise * self.noise_sigma
        costs = jax.vmap(self._rollout, in_axes=(None, 0))(state, actions)
        weights = jax.nn.softmax(-costs / self.temperature)
        best_action = jnp.einsum('n,nhd->hd', weights, actions)
        return best_action, costs


def _spearman_correlation(x: jax.Array, y: jax.Array) -> float:
    """Compute Spearman rank correlation."""
    rank_x = jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
    rank_y = jnp.argsort(jnp.argsort(y)).astype(jnp.float32)
    n = x.shape[0]
    mean_rx = jnp.mean(rank_x)
    mean_ry = jnp.mean(rank_y)
    cov = jnp.sum((rank_x - mean_rx) * (rank_y - mean_ry))
    std_x = jnp.sqrt(jnp.sum((rank_x - mean_rx) ** 2))
    std_y = jnp.sqrt(jnp.sum((rank_y - mean_ry) ** 2))
    return cov / (std_x * std_y + 1e-8)


def benchmark_clustering(
    dynamics_fn: Callable,
    cost_fn: Callable,
    state: jax.Array,
    key: jax.Array,
    k_values: list = [8, 16, 32, 64, 128],
    n_samples: int = 1024,
    n_trials: int = 10,
) -> list:
    """Benchmark clustering for various K values."""
    results = []

    for k in k_values:
        planner = ClusteredMPPI(
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            n_samples=n_samples,
            n_clusters=k,
            action_dim=2,
        )

        plan_cluster = jax.jit(planner.plan)
        plan_full = jax.jit(planner.plan_full)

        # Warm up
        key, k1, k2 = jax.random.split(key, 3)
        a1, c1 = plan_full(state, k1)
        jax.block_until_ready(a1)
        a2, c2 = plan_cluster(state, k2)
        jax.block_until_ready(a2)

        # Benchmark
        full_times, cluster_times = [], []
        rank_corrs, cost_mses = [], []

        for _ in range(n_trials):
            key, k1 = jax.random.split(key)

            t0 = time.perf_counter()
            _, true_costs = plan_full(state, k1)
            jax.block_until_ready(true_costs)
            full_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _, approx_costs = plan_cluster(state, k1)
            jax.block_until_ready(approx_costs)
            cluster_times.append(time.perf_counter() - t0)

            corr = _spearman_correlation(true_costs, approx_costs)
            mse = float(jnp.mean((true_costs - approx_costs) ** 2))
            rank_corrs.append(float(corr))
            cost_mses.append(mse)

        r = ClusterResult(
            k=k,
            full_sim_time=sum(full_times) / len(full_times),
            cluster_time=sum(cluster_times) / len(cluster_times),
            speedup=(sum(full_times) / len(full_times)) / (sum(cluster_times) / len(cluster_times)),
            rank_correlation=sum(rank_corrs) / len(rank_corrs),
            cost_mse=sum(cost_mses) / len(cost_mses),
        )
        results.append(r)

    return results


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
    print("=== B3-2: Trajectory Clustering Benchmark ===\n")

    state = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(42)

    results = benchmark_clustering(
        _dynamics, _cost, state, key,
        k_values=[8, 16, 32, 64, 128],
        n_samples=1024,
        n_trials=15,
    )

    print(f"{'K':>5} {'Full(ms)':>10} {'Cluster(ms)':>12} {'Speedup':>8} "
          f"{'Rank Corr':>10} {'Cost MSE':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.k:>5d} {r.full_sim_time*1000:>10.2f} {r.cluster_time*1000:>12.2f} "
              f"{r.speedup:>8.2f}x {r.rank_correlation:>10.4f} {r.cost_mse:>10.4f}")
