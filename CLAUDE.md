# Adaptive Fidelity Simulation & Parallel Rollout Research

## Environment
- Server: florence (8x RTX 3090, Ubuntu 20.04)
- Conda env: `robosim` (Python 3.10)
- Activate: `conda activate robosim`
- Packages: Genesis 0.4.2, JAX 0.6.2 (CUDA), PyTorch 2.6.0+cu124
- Video encoding: use `imageio` with `imageio_ffmpeg` (system ffmpeg is broken)

## Project Structure
```
~/robosim-research/
├── envs/
│   └── genesis/                 # Genesis unified physics engine
│       ├── fluid_sim.py         # MPM/SPH fluid sim with fidelity knobs
│       ├── manipulation.py      # Franka Panda manipulation (rigid body)
│       └── pouring.py           # Multi-phase pouring + adaptive fidelity
├── planning/
│   ├── mppi.py                  # JAX-native MPPI (jit-compiled, warm-start)
│   ├── cem.py                   # JAX-native CEM
│   └── reuse/                   # Parallel rollout reuse experiments
│       ├── shared_prefix.py     # B3-1: Share initial state computation
│       ├── trajectory_cluster.py # B3-2: K-means cluster + representative sim
│       ├── temporal_reuse.py    # B3-3: Cross-timestep reuse
│       └── two_stage.py         # B3-4: Coarse-to-fine two-stage eval
├── analysis/
│   ├── plots.py                 # Matplotlib plotting
│   └── video.py                 # MP4 video rendering (3 scenes: dam_break, franka, pouring)
└── videos/                      # Rendered videos
```

## Running Experiments
```bash
conda activate robosim
cd ~/robosim-research

# Genesis environments (GPU)
python -m envs.genesis.fluid_sim       # Fluid benchmark: LOW/MEDIUM/HIGH
python -m envs.genesis.manipulation    # Franka manipulation benchmark
python -m envs.genesis.pouring         # Pouring: fixed vs adaptive fidelity

# Video rendering
python -m analysis.video --scene all --output-dir videos/
python -m analysis.video --scene dam_break   # Single scene
```

## Research Directions

### Track A: Adaptive Fidelity (Genesis)
- `envs/genesis/fluid_sim.py`: MPM fluid, LOW/MEDIUM/HIGH presets (4ms / 8ms / 47ms per frame)
- `envs/genesis/pouring.py`: Multi-phase pouring with adaptive substep switching (1.7x speedup)
- Fidelity knobs: `particle_size`, `substeps`, `dt`, `grid_density`, solver type (MPM/SPH)
- Runtime substep switching: `scene.sim_options.substeps = N`

### Track B: Rigid-Body Manipulation (Genesis)
- `envs/genesis/manipulation.py`: 9-DOF Franka Panda, velocity control
- Fidelity knobs: `dt`, `substeps`, `solver_iterations`, `ls_iterations`
- LOW=2.5ms, MEDIUM=5ms, HIGH=11ms per step
- Built-in robot models: Franka, KUKA, Shadow Hand, Go2, ANYmal

### Track C: Integration (TODO)
- Combine adaptive fidelity + parallel rollout in Genesis
- Multi-physics scene: Franka arm + MPM fluid pouring
- Two-stage: low-fidelity coarse filter -> high-fidelity fine evaluation
- Genesis pouring scenario with MPPI control

## Known Issues / Genesis Gotchas
- System ffmpeg broken (libnppig.so.11 missing) -> always use imageio_ffmpeg
- Genesis warns about `torch<2.8.0` — harmless, works fine with 2.6.0+cu124
- Genesis MPM boundary has safety padding; keep fluid well within `lower_bound`/`upper_bound`
- Genesis URDF: use `fixed=True` for fixed-base robots (otherwise floating base adds 7 extra DOFs)
- Genesis API: use `get_dofs_position()`/`get_dofs_velocity()`, NOT `get_qvel()`
- Genesis `get_state().pos` returns shape `(1, N, 3)` with batch dim — use `.squeeze(0)` for `(N, 3)`
- Genesis `get_pos()` for rigid entity position (also has batch dim)
- `sim_options.dt` is total time per `scene.step()` (NOT per substep). Use dt=4e-3+ for visible fluid dynamics
- EGL context conflicts when creating multiple scenes in one process — use subprocess for multi-scene video rendering
- Camera: `scene.add_camera(res, pos, lookat, fov)` before build, then `cam.start_recording()`/`cam.render()`/`cam.stop_recording(save_to_filename, fps)`

## Conventions
- Genesis is the primary simulation backend for both Track A and B
- All JAX code should be jit-compatible (use jax.lax.scan, not Python loops)
- Each module has `if __name__ == "__main__"` for standalone testing
