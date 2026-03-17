# Adaptive Fidelity Simulation & Parallel Rollout Research

## Environment
- Primary server: **paris** (`~/simsim/`, 4x RTX A6000)
- Backup server: **florence** (`~/robosim-research/`, 8x RTX 3090) — GPU 0 broken, use `--gpu 2` or higher
- Conda env: `robosim` (Python 3.10)
- Activate: `conda activate robosim`
- Packages: Genesis 0.4.2, JAX 0.6.2 (CUDA), PyTorch 2.6.0+cu124
- GitHub: `yankee624/simsim`

## Project Structure
```
├── envs/
│   └── genesis/
│       ├── fluid_sim.py       # MPM fluid sim, FidelityConfig (LOW/MEDIUM/HIGH)
│       ├── manipulation.py    # Franka Panda 9-DOF, velocity control
│       ├── pouring.py         # Multi-phase pouring + AdaptiveFidelityController
│       ├── cup_carrying.py    # Kinematic cup + MPM fluid, spill metric for MPC
│       └── cloth.py           # IPC cloth (FEM.Cloth + IPCCoupler)
├── planning/
│   ├── mppi.py                # JAX-native MPPI (jit-compiled, warm-start)
│   ├── cem.py                 # JAX-native CEM
│   └── reuse/                 # Parallel rollout reuse experiments
│       ├── shared_prefix.py   # B3-1: Share initial state computation
│       ├── trajectory_cluster.py  # B3-2: K-means cluster + representative sim
│       ├── temporal_reuse.py  # B3-3: Cross-timestep reuse
│       └── two_stage.py       # B3-4: Coarse-to-fine two-stage eval
└── analysis/
    ├── plots.py               # Matplotlib plotting
    └── video.py               # MP4 rendering (dam_break, franka, pouring, cup_carry, cloth)
```

## Running Experiments
```bash
conda activate robosim
cd ~/simsim  # paris | or ~/robosim-research on florence

# Genesis environments
python -m envs.genesis.fluid_sim        # Fluid benchmark: LOW/MEDIUM/HIGH
python -m envs.genesis.manipulation     # Franka manipulation benchmark
python -m envs.genesis.pouring          # Pouring: fixed vs adaptive fidelity
python -m envs.genesis.cup_carrying     # Cup carry: spill fraction test

# Video rendering (each scene spawned as subprocess to avoid EGL conflicts)
python -m analysis.video --scene all --output-dir videos/
python -m analysis.video --scene cup_carry --gpu 1
```

## Simulation Modules

### fluid_sim.py — MPM Fluid
- Fidelity presets: LOW=4ms, MEDIUM=8ms, HIGH=47ms per frame
- Knobs: `particle_size`, `substeps`, `dt`, `grid_density`

### manipulation.py — Franka Panda
- 9-DOF velocity control; fidelity: LOW=2.5ms, MEDIUM=5ms, HIGH=11ms per step
- Knobs: `dt`, `substeps`, `solver_iterations`, `ls_iterations`

### pouring.py — Adaptive Fidelity Pouring
- Phase detection (TRANSPORT/POURING/SETTLING/STATIC) → `AdaptiveFidelityController`
- Runtime substep switching: `scene.sim_options.substeps = N` → 1.7x speedup over fixed HIGH

### cup_carrying.py — Cup + MPM Fluid (MPC target)
- Kinematic rigid cup (5 Box walls, `fixed=False` + `rho=1e6`) containing MPM liquid
- `rollout(trajectory) → {spill_fraction, max_spill, spill_history, mean_step_ms}`
- Fidelity: `CARRY_LOW` (ps=0.025), `CARRY_MEDIUM` (ps=0.012), `CARRY_HIGH` (ps=0.007)
- `CarryTrajectory`: `straight_line()`, `random_walk()`, `figure_eight()`
- Next step: wire up MPPI/CEM to minimize spill_fraction

### cloth.py — IPC Cloth
- FEM.Cloth + IPCCoupler; fidelity: LOW=163ms, MEDIUM=487ms, HIGH=930ms per frame
- Requires cublas preload before Genesis import (done inside `render_cloth()` only)

## Known Issues / Genesis Gotchas

### MPM-Rigid Coupling (critical)
- `fixed=True` walls have velocity=0 in Genesis → MPM treats as static, fluid not pushed
- Fix: `fixed=False` + `rho=1e6`, call `set_pos()` + `set_dofs_velocity([vx,vy,vz,0,0,0])` each step
- Wall thickness must be ≥ 2× particle_size or MPM grid can't detect it → fluid tunnels through
- Safe thickness: `max(2.5 * particle_size, 0.015)`

### API
- Use `get_dofs_position()` / `get_dofs_velocity()`, NOT `get_qvel()`
- `get_state().pos` returns `(1, N, 3)` — use `.squeeze(0)` for `(N, 3)`
- `sim_options.dt` = total time per `scene.step()` (NOT per substep); use dt ≥ 4e-3 for visible fluid
- URDF: `fixed=True` for fixed-base robots (otherwise floating base adds 7 extra DOFs)

### Infrastructure
- System ffmpeg broken on florence (libnppig.so.11 missing) → always use `imageio` + `imageio_ffmpeg`
- Genesis warns `torch<2.8.0` — harmless with 2.6.0+cu124
- EGL context conflicts when creating multiple scenes in one process → use subprocess (video.py already does this)
- Camera: `scene.add_camera(res, pos, lookat, fov)` before build → `cam.start_recording()` → `cam.render()` per step → `cam.stop_recording(save_to_filename, fps)`
- IPC cublas preload (`ctypes.CDLL(libcublas.so.12, RTLD_GLOBAL)`) must be inside `render_cloth()` only — loading it globally breaks CUDA init for other scenes

## Conventions
- All JAX code must be jit-compatible: use `jax.lax.scan`, not Python loops
- Each module has `if __name__ == "__main__"` for standalone testing
