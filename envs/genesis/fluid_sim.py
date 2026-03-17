"""Genesis-based fluid simulation with explicit fidelity controls.

GPU-accelerated MPM/SPH fluid simulation with tunable fidelity:
  - particle_size: spatial resolution (smaller = more particles = higher fidelity)
  - substeps: physics substeps per scene step
  - solver type: MPM vs SPH (MPM is faster, SPH more accurate for free-surface)
  - grid_density: MPM background grid resolution

All fidelity parameters can be changed by rebuilding the scene,
and substeps can be changed at runtime via scene.sim_options.
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import genesis as gs


# ============================================================
# Fidelity Configuration
# ============================================================

class SolverType(Enum):
    MPM = "mpm"
    SPH = "sph"


@dataclass
class FidelityConfig:
    """Tunable fidelity parameters for adaptive simulation."""
    particle_size: float = 0.015       # spatial resolution
    substeps: int = 10                 # physics substeps per scene step
    solver_type: SolverType = SolverType.MPM
    grid_density: float = 64.0         # MPM grid density (higher = finer)
    dt: float = 5e-4                   # simulation timestep

    @property
    def label(self) -> str:
        return (f"solver={self.solver_type.value},ps={self.particle_size},"
                f"sub={self.substeps},dt={self.dt}")


# Presets
FIDELITY_LOW = FidelityConfig(
    particle_size=0.03, substeps=5, grid_density=32.0, dt=1e-3,
)
FIDELITY_MEDIUM = FidelityConfig(
    particle_size=0.015, substeps=10, grid_density=64.0, dt=5e-4,
)
FIDELITY_HIGH = FidelityConfig(
    particle_size=0.008, substeps=20, grid_density=128.0, dt=2e-4,
)


# ============================================================
# Fluid Simulation
# ============================================================

class FluidSimulation:
    """Genesis fluid simulation with runtime fidelity control.

    Usage:
        sim = FluidSimulation(fidelity=FIDELITY_MEDIUM)
        sim.build_dam_break()
        for _ in range(100):
            sim.step()
        pos = sim.get_positions()
    """

    def __init__(
        self,
        fidelity: Optional[FidelityConfig] = None,
        device_id: int = 0,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        bound_min: Tuple[float, float, float] = (-1.0, -1.0, -0.5),
        bound_max: Tuple[float, float, float] = (1.0, 1.0, 1.5),
        camera: Optional[dict] = None,
    ):
        self.fidelity = fidelity or FIDELITY_MEDIUM
        self.device_id = device_id
        self.gravity = gravity
        self.bound_min = bound_min
        self.bound_max = bound_max
        self._camera_cfg = camera

        self.scene: Optional[gs.Scene] = None
        self.fluid_entity = None
        self.cam = None
        self.n_particles = 0
        self.frame_count = 0
        self._step_times = []

    def _make_solver_options(self):
        f = self.fidelity
        if f.solver_type == SolverType.MPM:
            return gs.options.solvers.MPMOptions(
                dt=f.dt,
                particle_size=f.particle_size,
                grid_density=f.grid_density,
                lower_bound=self.bound_min,
                upper_bound=self.bound_max,
            )
        else:
            return gs.options.solvers.SPHOptions(
                dt=f.dt,
                particle_size=f.particle_size,
                lower_bound=self.bound_min,
                upper_bound=self.bound_max,
            )

    def _make_material(self):
        if self.fidelity.solver_type == SolverType.MPM:
            return gs.materials.MPM.Liquid(
                sampler="pbs",
            )
        else:
            return gs.materials.SPH.Liquid(
                sampler="pbs",
            )

    def build_dam_break(
        self,
        fluid_pos: Tuple[float, float, float] = (-0.2, -0.2, 0.25),
        fluid_size: Tuple[float, float, float] = (0.3, 0.3, 0.3),
        walls: bool = False,
        obstacle: bool = False,
        fluid_color: Optional[Tuple] = None,
    ):
        """Build a dam break scene: block of fluid released in a container."""
        f = self.fidelity
        solver_options = self._make_solver_options()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=f.dt,
                substeps=f.substeps,
                gravity=self.gravity,
            ),
            show_viewer=False,
            rigid_options=gs.options.solvers.RigidOptions(
                dt=f.dt,
                enable_collision=True,
            ),
            mpm_options=solver_options if f.solver_type == SolverType.MPM else None,
            sph_options=solver_options if f.solver_type == SolverType.SPH else None,
        )

        # Ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Optional containment wall (the dam)
        if walls:
            wall_surface = gs.surfaces.Default(color=(0.55, 0.5, 0.45, 1.0))
            self.scene.add_entity(
                gs.morphs.Box(pos=(-0.6, 0.0, 0.3), size=(0.02, 0.8, 0.6), fixed=True),
                surface=wall_surface,
            )

        # Optional obstacle in flow path
        if obstacle:
            self.scene.add_entity(
                gs.morphs.Box(pos=(0.3, 0.0, 0.06), size=(0.12, 0.12, 0.12), fixed=True),
                surface=gs.surfaces.Default(color=(0.8, 0.3, 0.2, 1.0)),
            )

        # Fluid block
        fluid_kwargs = dict(
            morph=gs.morphs.Box(pos=fluid_pos, size=fluid_size),
            material=self._make_material(),
        )
        if fluid_color:
            fluid_kwargs["surface"] = gs.surfaces.Default(
                color=fluid_color, vis_mode="particle",
            )
        self.fluid_entity = self.scene.add_entity(**fluid_kwargs)

        # Camera (must be added before build)
        if self._camera_cfg:
            self.cam = self.scene.add_camera(**self._camera_cfg)

        self.scene.build()
        self.n_particles = self.fluid_entity.n_particles
        self.frame_count = 0
        self._step_times = []

        return self.n_particles

    def step(self):
        """Advance simulation by one scene step."""
        t0 = time.perf_counter()
        self.scene.step()
        elapsed = time.perf_counter() - t0
        self.frame_count += 1
        self._step_times.append(elapsed)

    def get_positions(self) -> np.ndarray:
        """Get particle positions as (N, 3) numpy array."""
        return self.fluid_entity.get_state().pos.cpu().numpy().squeeze(0)

    def get_velocities(self) -> np.ndarray:
        """Get particle velocities as (N, 3) numpy array."""
        return self.fluid_entity.get_state().vel.cpu().numpy().squeeze(0)

    def get_max_speed(self) -> float:
        """Get maximum particle speed."""
        vel = self.get_velocities()
        return float(np.max(np.linalg.norm(vel, axis=1)))

    def get_kinetic_energy(self) -> float:
        """Get total kinetic energy (approximate, uniform mass)."""
        vel = self.get_velocities()
        speed_sq = np.sum(vel ** 2, axis=1)
        # Approximate mass per particle from density and volume
        vol_per_particle = self.fidelity.particle_size ** 3
        mass = 1000.0 * vol_per_particle  # water density
        return float(0.5 * mass * np.sum(speed_sq))

    def get_step_stats(self) -> dict:
        if not self._step_times:
            return {}
        times = np.array(self._step_times)
        return {
            "mean_step_ms": float(times.mean() * 1000),
            "std_step_ms": float(times.std() * 1000),
            "min_step_ms": float(times.min() * 1000),
            "max_step_ms": float(times.max() * 1000),
            "total_frames": self.frame_count,
            "particles": self.n_particles,
            "fidelity": self.fidelity.label,
        }

    # ---- Camera / recording ----

    def render_frame(self):
        if self.cam:
            self.cam.render()

    def start_recording(self):
        if self.cam:
            self.cam.start_recording()

    def stop_recording(self, filename: str, fps: int = 30):
        if self.cam:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            self.cam.stop_recording(save_to_filename=filename, fps=fps)


# ============================================================
# CLI: benchmark fidelity levels
# ============================================================

if __name__ == "__main__":
    gs.init(backend=gs.cuda, logging_level="warning")

    print("=== Genesis Fluid Simulation Benchmark ===\n")

    for name, fidelity in [("LOW", FIDELITY_LOW), ("MEDIUM", FIDELITY_MEDIUM), ("HIGH", FIDELITY_HIGH)]:
        sim = FluidSimulation(fidelity=fidelity)
        n = sim.build_dam_break()
        print(f"[{name}] {fidelity.label}")
        print(f"  Particles: {n}")

        # Warm-up
        sim.step()

        # Benchmark
        n_frames = 50
        t0 = time.perf_counter()
        for _ in range(n_frames):
            sim.step()
        elapsed = time.perf_counter() - t0

        stats = sim.get_step_stats()
        max_v = sim.get_max_speed()
        ke = sim.get_kinetic_energy()

        print(f"  {n_frames} frames in {elapsed:.2f}s ({elapsed/n_frames*1000:.1f} ms/frame)")
        print(f"  max_speed={max_v:.3f}, KE={ke:.4f}")
        print(f"  Stats: {stats['mean_step_ms']:.1f} +/- {stats['std_step_ms']:.1f} ms/frame")
        print()
