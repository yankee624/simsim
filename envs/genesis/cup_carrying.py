"""Cup-carrying fluid simulation for MPC trajectory evaluation.

A rigid cup filled with MPM liquid is moved along a trajectory.
The cost metric is the fraction of liquid that spills out.

MPC usage pattern:
    sim = CupCarryingSim(fidelity=CARRY_MEDIUM)
    sim.build()
    for traj in candidate_trajectories:          # N trajectories in parallel
        result = sim.rollout(traj)
        cost = result["spill_fraction"]
    best = min(results, key=lambda r: r["spill_fraction"])
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import genesis as gs


# ============================================================
# Fidelity presets
# ============================================================

@dataclass
class CarryFidelityConfig:
    particle_size: float = 0.012
    substeps: int = 10
    dt: float = 5e-4
    grid_density: float = 64.0

    @property
    def label(self) -> str:
        return f"ps={self.particle_size},sub={self.substeps},dt={self.dt}"


CARRY_LOW    = CarryFidelityConfig(particle_size=0.025, substeps=5,  dt=1e-3,  grid_density=32.0)
CARRY_MEDIUM = CarryFidelityConfig(particle_size=0.012, substeps=10, dt=5e-4,  grid_density=64.0)
CARRY_HIGH   = CarryFidelityConfig(particle_size=0.007, substeps=20, dt=2e-4,  grid_density=128.0)


# ============================================================
# Cup geometry helpers
# ============================================================

# Cup dimensions (all in metres)
CUP_RADIUS = 0.06   # inner half-width (square cup)
CUP_HEIGHT = 0.12   # inner height
CUP_FILL   = 0.55   # fraction of inner height filled with fluid


def _cup_wall_thickness(particle_size: float) -> float:
    """Wall thickness must be >= 2× MPM grid cell (≈ particle_size) so the
    MPM solver can detect the rigid body boundary correctly."""
    return max(2.5 * particle_size, 0.015)


def _cup_walls(cx: float, cy: float, cz_bottom: float, th: float):
    """Return list of (pos, size) for 5 rigid boxes forming the cup.

    cz_bottom: Z of the outer bottom surface.
    th: wall thickness (must be >= 2 × particle_size for MPM coupling).
    """
    r = CUP_RADIUS
    h = CUP_HEIGHT
    # bottom plate
    bottom_pos  = (cx, cy, cz_bottom + th / 2)
    bottom_size = (2*(r + th), 2*(r + th), th)
    # four walls (inner face at ±r from centre)
    wall_h  = h + th   # full height including bottom
    wall_cz = cz_bottom + th + h / 2
    walls = [
        # +x
        ((cx + r + th/2, cy, wall_cz), (th, 2*(r + th), wall_h)),
        # -x
        ((cx - r - th/2, cy, wall_cz), (th, 2*(r + th), wall_h)),
        # +y
        ((cx, cy + r + th/2, wall_cz), (2*(r + th), th, wall_h)),
        # -y
        ((cx, cy - r - th/2, wall_cz), (2*(r + th), th, wall_h)),
    ]
    return [(bottom_pos, bottom_size)] + walls


def _fluid_init_pos(cx: float, cy: float, cz_bottom: float, th: float):
    """Centre of initial fluid block inside cup."""
    h = CUP_HEIGHT * CUP_FILL
    return (cx, cy, cz_bottom + th + h / 2)


def _fluid_init_size():
    r = CUP_RADIUS * 0.80   # 80% of inner radius — clear of walls
    h = CUP_HEIGHT * CUP_FILL
    return (2 * r, 2 * r, h)


# ============================================================
# Trajectory dataclass
# ============================================================

@dataclass
class CarryTrajectory:
    """Sequence of cup (dx, dy, dz) displacements per scene step."""
    deltas: np.ndarray    # shape (T, 3)  — displacement per step

    @staticmethod
    def straight_line(
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        n_steps: int,
    ) -> "CarryTrajectory":
        start = np.array(start)
        end   = np.array(end)
        delta = (end - start) / n_steps
        return CarryTrajectory(deltas=np.tile(delta, (n_steps, 1)))

    @staticmethod
    def random_walk(
        n_steps: int,
        max_speed: float = 0.003,
        seed: Optional[int] = None,
    ) -> "CarryTrajectory":
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-max_speed, max_speed, (n_steps, 3))
        raw[:, 2] = 0.0    # no vertical movement by default
        return CarryTrajectory(deltas=raw)

    @staticmethod
    def figure_eight(n_steps: int, radius: float = 0.08) -> "CarryTrajectory":
        t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
        x = radius * np.sin(t)
        y = radius * np.sin(t) * np.cos(t)
        dx = np.diff(x, append=x[0:1])
        dy = np.diff(y, append=y[0:1])
        deltas = np.stack([dx, dy, np.zeros(n_steps)], axis=1)
        return CarryTrajectory(deltas=deltas)


# ============================================================
# Cup-carrying simulation
# ============================================================

class CupCarryingSim:
    """Rigid cup + MPM fluid simulation.

    The cup is kinematic: at each step its position is updated directly.
    Spill is measured by counting particles outside the cup footprint + height.
    """

    def __init__(
        self,
        fidelity: Optional[CarryFidelityConfig] = None,
        cup_init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        camera: Optional[dict] = None,
        bound_min: Tuple[float, float, float] = (-0.8, -0.8, -0.1),
        bound_max: Tuple[float, float, float] = (0.8,  0.8,  0.8),
    ):
        self.fidelity     = fidelity or CARRY_MEDIUM
        self.cup_init_pos = np.array(cup_init_pos, dtype=float)
        self._camera_cfg  = camera
        self.bound_min    = bound_min
        self.bound_max    = bound_max

        self.scene: Optional[gs.Scene] = None
        self.fluid        = None
        self.cup_parts: List = []
        self.cam          = None
        self.n_particles  = 0
        self._cup_pos     = self.cup_init_pos.copy()
        self._step_times: List[float] = []
        self.frame_count  = 0

    # ------------------------------------------------------------------ build

    def build(self, fluid_color: Optional[Tuple] = None):
        f = self.fidelity
        cx, cy, cz = self.cup_init_pos
        th = _cup_wall_thickness(f.particle_size)
        self._wall_th = th

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=f.dt,
                substeps=f.substeps,
                gravity=(0.0, 0.0, -9.81),
            ),
            mpm_options=gs.options.solvers.MPMOptions(
                dt=f.dt,
                particle_size=f.particle_size,
                grid_density=f.grid_density,
                lower_bound=self.bound_min,
                upper_bound=self.bound_max,
                enable_CPIC=True,
            ),
            rigid_options=gs.options.solvers.RigidOptions(
                dt=f.dt,
                enable_collision=True,
            ),
            show_viewer=False,
        )

        # Ground
        self.scene.add_entity(gs.morphs.Plane())

        # Cup walls: non-fixed with rho=1e6 so fluid forces can't move them.
        # We control them kinematically: set_pos() + set_dofs_velocity() each step
        # so the MPM solver sees both correct position AND velocity boundary condition.
        # Wall thickness >= 2.5 × particle_size ensures MPM grid detects the wall.
        cup_color = (0.75, 0.75, 0.80, 0.7)
        self.cup_parts = []
        for pos, size in _cup_walls(cx, cy, cz, th):
            part = self.scene.add_entity(
                gs.morphs.Box(pos=pos, size=size, fixed=False),
                material=gs.materials.Rigid(rho=1e6),
                surface=gs.surfaces.Default(color=cup_color),
            )
            self.cup_parts.append(part)

        # Fluid
        fpos = _fluid_init_pos(cx, cy, cz, th)
        fsz  = _fluid_init_size()
        fluid_kwargs = dict(
            morph=gs.morphs.Box(pos=fpos, size=fsz),
            material=gs.materials.MPM.Liquid(sampler="pbs"),
        )
        if fluid_color:
            fluid_kwargs["surface"] = gs.surfaces.Default(
                color=fluid_color, vis_mode="particle",
            )
        self.fluid = self.scene.add_entity(**fluid_kwargs)

        if self._camera_cfg:
            self.cam = self.scene.add_camera(**self._camera_cfg)

        self.scene.build()
        self.n_particles  = self.fluid.n_particles
        self._cup_pos     = self.cup_init_pos.copy()
        self._step_times  = []
        self.frame_count  = 0
        return self.n_particles

    # ------------------------------------------------------------------ step

    def step(self, delta: Optional[np.ndarray] = None):
        """Advance one scene step, optionally moving the cup by `delta` (m)."""
        if delta is not None:
            self._move_cup(np.array(delta, dtype=float))
        t0 = time.perf_counter()
        self.scene.step()
        self._step_times.append(time.perf_counter() - t0)
        self.frame_count += 1

    def _move_cup(self, delta: np.ndarray):
        """Move cup kinematically: set position + velocity so MPM coupling works.

        set_pos()           → corrects position each step (prevents drift)
        set_dofs_velocity() → tells MPM solver the boundary velocity so fluid
                              gets pushed instead of being treated as a static wall
        """
        self._cup_pos += delta
        # Velocity = displacement / scene step duration
        scene_step_dt = self.fidelity.dt * self.fidelity.substeps
        v = delta / max(scene_step_dt, 1e-9)
        dof_vel = np.array([v[0], v[1], v[2], 0.0, 0.0, 0.0])

        new_geoms = _cup_walls(*self._cup_pos, self._wall_th)
        for part, (pos, _) in zip(self.cup_parts, new_geoms):
            part.set_pos(np.array(pos))
            part.set_dofs_velocity(dof_vel)

    # ------------------------------------------------------------------ metrics

    def get_positions(self) -> np.ndarray:
        """Particle positions (N, 3)."""
        return self.fluid.get_state().pos.cpu().numpy().squeeze(0)

    def get_spill_count(self) -> int:
        """Number of particles outside the cup."""
        pos = self.get_positions()
        return int(np.sum(~self._inside_cup(pos)))

    def get_spill_fraction(self) -> float:
        if self.n_particles == 0:
            return 0.0
        return self.get_spill_count() / self.n_particles

    def _inside_cup(self, pos: np.ndarray) -> np.ndarray:
        """Boolean mask: True if particle is geometrically inside cup."""
        cx, cy, cz = self._cup_pos
        th = self._wall_th
        r  = CUP_RADIUS + th          # outer edge
        z0 = cz                       # bottom outer surface
        z1 = cz + th + CUP_HEIGHT
        inside_x = np.abs(pos[:, 0] - cx) < r
        inside_y = np.abs(pos[:, 1] - cy) < r
        inside_z = (pos[:, 2] > z0) & (pos[:, 2] < z1 + 0.02)
        return inside_x & inside_y & inside_z

    # ------------------------------------------------------------------ rollout (MPC)

    def rollout(
        self,
        trajectory: CarryTrajectory,
        record_video: bool = False,
        video_path: Optional[str] = None,
    ) -> dict:
        """Execute a full trajectory and return cost metrics.

        Returns:
            dict with keys:
              spill_fraction   - fraction of particles outside cup at end
              spill_count      - absolute count
              max_spill        - peak spill fraction during rollout
              mean_step_ms     - wall-clock ms per step
              final_cup_pos    - (x, y, z) of cup at end
              spill_history    - list of spill_fraction per step
        """
        if record_video and self.cam:
            self.cam.start_recording()

        spill_history = []
        for delta in trajectory.deltas:
            self.step(delta)
            spill_history.append(self.get_spill_fraction())
            if record_video and self.cam:
                self.cam.render()

        if record_video and self.cam and video_path:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self.cam.stop_recording(save_to_filename=video_path, fps=30)

        times = np.array(self._step_times[-len(trajectory.deltas):])
        return {
            "spill_fraction":  float(spill_history[-1]) if spill_history else 0.0,
            "spill_count":     self.get_spill_count(),
            "max_spill":       float(max(spill_history)) if spill_history else 0.0,
            "mean_step_ms":    float(times.mean() * 1000) if len(times) else 0.0,
            "final_cup_pos":   self._cup_pos.tolist(),
            "spill_history":   spill_history,
        }

    # ------------------------------------------------------------------ camera

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

    def get_step_stats(self) -> dict:
        if not self._step_times:
            return {}
        t = np.array(self._step_times)
        return {
            "mean_step_ms": float(t.mean() * 1000),
            "std_step_ms":  float(t.std() * 1000),
            "total_frames": self.frame_count,
            "particles":    self.n_particles,
            "fidelity":     self.fidelity.label,
        }


# ============================================================
# CLI: benchmark + MPC demo
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "mpc"], default="benchmark")
    parser.add_argument("--fidelity", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--n-candidates", type=int, default=5, help="MPC candidate trajectories")
    parser.add_argument("--n-steps", type=int, default=120)
    args = parser.parse_args()

    gs.init(backend=gs.cuda, logging_level="warning")

    fid_map = {"low": CARRY_LOW, "medium": CARRY_MEDIUM, "high": CARRY_HIGH}
    fidelity = fid_map[args.fidelity]

    if args.mode == "benchmark":
        print("=== Cup-Carrying Fluid Benchmark ===\n")
        for name, fid in [("LOW", CARRY_LOW), ("MEDIUM", CARRY_MEDIUM), ("HIGH", CARRY_HIGH)]:
            sim = CupCarryingSim(fidelity=fid)
            n = sim.build()
            print(f"[{name}] {fid.label}  |  particles={n}")
            # warm-up
            sim.step()
            # benchmark 50 steps with gentle rocking
            n_steps = 50
            t0 = time.perf_counter()
            for i in range(n_steps):
                delta = np.array([0.001 * np.sin(i * 0.2), 0.0, 0.0])
                sim.step(delta)
            elapsed = time.perf_counter() - t0
            spill = sim.get_spill_fraction()
            print(f"  {n_steps} steps in {elapsed:.2f}s  ({elapsed/n_steps*1000:.1f} ms/step)")
            print(f"  spill fraction: {spill:.4f}\n")

    elif args.mode == "mpc":
        print(f"=== MPC Trajectory Selection Demo ({args.n_candidates} candidates) ===\n")
        print("Generating candidate trajectories...")
        candidates = [
            CarryTrajectory.straight_line((0, 0, 0), (0.3, 0, 0), args.n_steps),
            CarryTrajectory.straight_line((0, 0, 0), (0, 0.3, 0), args.n_steps),
            CarryTrajectory.figure_eight(args.n_steps, radius=0.06),
        ] + [
            CarryTrajectory.random_walk(args.n_steps, max_speed=0.002, seed=i)
            for i in range(args.n_candidates - 3)
        ]

        results = []
        for i, traj in enumerate(candidates):
            sim = CupCarryingSim(fidelity=fidelity)
            sim.build()
            result = sim.rollout(traj)
            result["traj_idx"] = i
            results.append(result)
            print(f"  Traj {i}: spill={result['spill_fraction']:.4f}  "
                  f"max_spill={result['max_spill']:.4f}  "
                  f"{result['mean_step_ms']:.1f} ms/step")

        best = min(results, key=lambda r: r["spill_fraction"])
        print(f"\nBest trajectory: #{best['traj_idx']}  "
              f"spill={best['spill_fraction']:.4f}")
        print(f"Final cup pos: {best['final_cup_pos']}")
