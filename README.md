# Adaptive Fidelity Simulation & Parallel Rollout Reuse

Research environment for studying online robot adaptation through simulation.

## Tracks

### Track A: Continuous Simulation (FluidLab + Taichi)
- **A2-1**: Adaptive timestep (CFL-based dt adjustment)
- **A2-2**: Particle freeze/wake (skip static particles)
- **A2-3**: Physics engine switching (simplified model for transport phase)

### Track B: Rigid-Body Simulation (MuJoCo MJX + JAX)
- **B3-1**: Shared prefix computation across parallel rollouts
- **B3-2**: Trajectory clustering (simulate K representatives)
- **B3-3**: Temporal reuse across MPC timesteps
- **B3-4**: Two-stage coarse-to-fine evaluation

## Quick Start

```bash
conda activate robosim

# Track A experiments (NumPy-only, no GPU needed)
cd ~/robosim-research
python -m experiments.adaptive_fidelity --experiment all

# Track B experiments (requires JAX + MuJoCo)
python -m experiments.parallel_reuse --experiment all

# Baselines
python -m experiments.benchmark_baseline --track all

# Generate plots from results
python -m analysis.plots results.json --output-dir plots/
```

## Structure

```
envs/fluidlab/     # Track A: adaptive fidelity experiments
envs/mjx/          # Track B: MuJoCo MJX environments
planning/          # MPPI, CEM planners
planning/reuse/    # B3-1 through B3-4 reuse experiments
experiments/       # Experiment runners
analysis/          # Plotting utilities
```

## Dependencies

```
jax[cuda12], mujoco, mujoco-mjx, taichi, numpy, matplotlib, scipy
```
