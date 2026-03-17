"""Plotting utilities for experiment results.

Generates figures for:
- Throughput benchmarks
- Adaptive fidelity Pareto curves
- Rollout reuse comparisons
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def plot_mjx_throughput(results: Dict, save_path: str = "mjx_throughput.png"):
    """Plot MJX simulation throughput vs batch size."""
    if not HAS_MPL:
        return

    batch_sizes = sorted(int(k) for k in results.keys())
    env_sps = [results[str(n)]["env_steps_per_sec"] for n in batch_sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batch_sizes, env_sps, 'bo-', linewidth=2, markersize=8)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Batch Size (number of environments)')
    ax.set_ylabel('Environment Steps / Second')
    ax.set_title('MJX Simulation Throughput')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(n) for n in batch_sizes])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_adaptive_dt_pareto(results: Dict, save_path: str = "adaptive_dt_pareto.png"):
    """Plot speedup vs accuracy for adaptive timestep."""
    if not HAS_MPL:
        return

    cfls = sorted(float(k) for k in results.keys())
    speedups = [results[str(c)]["speedup"] for c in cfls]
    mses = [results[str(c)]["mse"] for c in cfls]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(speedups, mses, c=cfls, cmap='viridis', s=100, zorder=5)
    for i, cfl in enumerate(cfls):
        ax.annotate(f'CFL={cfl}', (speedups[i], mses[i]),
                    textcoords="offset points", xytext=(10, 5))

    ax.set_xlabel('Speedup (x)')
    ax.set_ylabel('State MSE')
    ax.set_yscale('log')
    ax.set_title('Adaptive Timestep: Speedup vs Accuracy')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='CFL Factor')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_freeze_analysis(results: Dict, save_path: str = "freeze_analysis.png"):
    """Plot freeze ratio / accuracy / speedup trade-off."""
    if not HAS_MPL:
        return

    thresholds = sorted(float(k) for k in results.keys())
    freeze_ratios = [results[str(t)]["freeze_ratio"] * 100 for t in thresholds]
    mses = [results[str(t)]["mse"] for t in thresholds]
    speedups = [results[str(t)]["speedup"] for t in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogx(thresholds, freeze_ratios, 'ro-')
    axes[0].set_xlabel('Velocity Threshold')
    axes[0].set_ylabel('Average Freeze Ratio (%)')
    axes[0].set_title('Freeze Ratio vs Threshold')
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(thresholds, mses, 'bs-')
    axes[1].set_xlabel('Velocity Threshold')
    axes[1].set_ylabel('State MSE')
    axes[1].set_title('Accuracy vs Threshold')
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(thresholds, speedups, 'g^-')
    axes[2].set_xlabel('Velocity Threshold')
    axes[2].set_ylabel('Speedup (x)')
    axes[2].set_title('Speedup vs Threshold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_clustering_tradeoff(results: Dict, save_path: str = "clustering_tradeoff.png"):
    """Plot K vs speedup vs rank correlation."""
    if not HAS_MPL:
        return

    ks = sorted(int(k) for k in results.keys())
    speedups = [results[str(k)]["speedup"] for k in ks]
    rank_corrs = [results[str(k)]["rank_correlation"] for k in ks]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = 'tab:blue'
    ax1.plot(ks, speedups, 'o-', color=color1, linewidth=2, label='Speedup')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Speedup (x)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(ks, rank_corrs, 's--', color=color2, linewidth=2, label='Rank Correlation')
    ax2.set_ylabel('Spearman Rank Correlation', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Trajectory Clustering: K vs Performance')
    ax1.grid(True, alpha=0.3)
    fig.legend(loc='lower right', bbox_to_anchor=(0.85, 0.15))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_reuse_comparison(
    prefix_results: Optional[Dict] = None,
    cluster_results: Optional[Dict] = None,
    temporal_results: Optional[Dict] = None,
    twostage_results: Optional[Dict] = None,
    save_path: str = "reuse_comparison.png",
):
    """Summary bar chart comparing all reuse strategies."""
    if not HAS_MPL:
        return

    strategies = []
    speedups = []

    if prefix_results and "1024" in prefix_results:
        strategies.append("Shared Prefix")
        speedups.append(prefix_results["1024"]["speedup"])

    if cluster_results and "32" in cluster_results:
        strategies.append("Clustering\n(K=32)")
        speedups.append(cluster_results["32"]["speedup"])

    if temporal_results and "512" in temporal_results:
        strategies.append("Temporal\nReuse")
        speedups.append(temporal_results["512"]["speedup"])

    if twostage_results:
        strategies.append("Two-Stage")
        speedups.append(twostage_results.get("speedup", 1.0))

    if not strategies:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax.bar(strategies, speedups, color=colors[:len(strategies)], width=0.6)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Parallel Rollout Reuse Strategies Comparison')

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(results_file: str, output_dir: str = "."):
    """Generate all plots from a JSON results file."""
    with open(results_file) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    if "mjx_throughput" in data:
        plot_mjx_throughput(data["mjx_throughput"], str(out / "mjx_throughput.png"))

    if "adaptive_timestep" in data:
        plot_adaptive_dt_pareto(data["adaptive_timestep"], str(out / "adaptive_dt_pareto.png"))

    if "particle_freeze" in data:
        plot_freeze_analysis(data["particle_freeze"], str(out / "freeze_analysis.png"))

    if "trajectory_clustering" in data:
        plot_clustering_tradeoff(
            data["trajectory_clustering"], str(out / "clustering_tradeoff.png"))

    # Summary comparison
    plot_reuse_comparison(
        prefix_results=data.get("shared_prefix"),
        cluster_results=data.get("trajectory_clustering"),
        temporal_results=data.get("temporal_reuse"),
        twostage_results=data.get("two_stage"),
        save_path=str(out / "reuse_comparison.png"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="JSON results file from experiments")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    args = parser.parse_args()

    generate_all_plots(args.results_file, args.output_dir)
