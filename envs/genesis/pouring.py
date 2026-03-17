"""Multi-phase pouring scenario: rigid robot + fluid in one scene.

A container filled with liquid is transported and tilted to pour
into a target region. Genesis handles rigid-fluid coupling natively.

Phases:
  1. TRANSPORT  - container moves to pouring position (low fidelity OK)
  2. POURING    - container tilts, fluid flows out (high fidelity needed)
  3. SETTLING   - fluid decelerates in target (medium -> low)
  4. STATIC     - everything at rest (skip or minimal computation)

Adaptive fidelity switches substeps/dt at phase boundaries.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import genesis as gs


# ============================================================
# Phase Detection
# ============================================================

class SimPhase(Enum):
    TRANSPORT = "transport"
    POURING = "pouring"
    SETTLING = "settling"
    STATIC = "static"


@dataclass
class PhaseThresholds:
    pour_angle: float = 0.3
    settling_speed: float = 0.5
    static_speed: float = 0.01
    static_energy: float = 1e-5


class PhaseDetector:
    def __init__(self, thresholds: Optional[PhaseThresholds] = None):
        self.thresholds = thresholds or PhaseThresholds()
        self.history: List[SimPhase] = []

    def detect(
        self,
        container_speed: float,
        container_angle: float,
        fluid_max_speed: float,
        fluid_ke: float,
    ) -> SimPhase:
        t = self.thresholds
        if abs(container_angle) > t.pour_angle:
            phase = SimPhase.POURING
        elif container_speed > t.static_speed:
            phase = SimPhase.TRANSPORT
        elif fluid_max_speed > t.settling_speed:
            phase = SimPhase.SETTLING
        elif fluid_ke < t.static_energy:
            phase = SimPhase.STATIC
        else:
            phase = SimPhase.SETTLING
        self.history.append(phase)
        return phase


# ============================================================
# Fidelity Policy
# ============================================================

@dataclass
class SubstepPolicy:
    """Maps phases to substep counts (runtime fidelity knob)."""
    transport: int = 5
    pouring: int = 20
    settling: int = 10
    static: int = 1

    def get(self, phase: SimPhase) -> int:
        return {
            SimPhase.TRANSPORT: self.transport,
            SimPhase.POURING: self.pouring,
            SimPhase.SETTLING: self.settling,
            SimPhase.STATIC: self.static,
        }[phase]


# ============================================================
# Container Trajectory (scripted motion)
# ============================================================

@dataclass
class ContainerTrajectory:
    total_frames: int = 600
    transport_end: int = 150
    pour_end: int = 350
    hold_end: int = 450
    return_end: int = 550
    transport_speed: float = 0.002  # displacement per frame
    max_pour_angle: float = 1.2     # rad

    def get_state(self, frame: int) -> Tuple[np.ndarray, float, float]:
        """Returns (displacement_xyz, angle_rad, container_speed)."""
        if frame < self.transport_end:
            dx = self.transport_speed
            disp = np.array([dx, 0.0, 0.0])
            angle = 0.0
            speed = abs(dx)
        elif frame < self.pour_end:
            t = (frame - self.transport_end) / (self.pour_end - self.transport_end)
            disp = np.array([0.0, 0.0, 0.0])
            angle = self.max_pour_angle * min(t * 1.5, 1.0)
            speed = 0.0
        elif frame < self.hold_end:
            disp = np.array([0.0, 0.0, 0.0])
            angle = self.max_pour_angle
            speed = 0.0
        elif frame < self.return_end:
            t = (frame - self.hold_end) / (self.return_end - self.hold_end)
            disp = np.array([0.0, 0.0, 0.0])
            angle = self.max_pour_angle * (1.0 - t)
            speed = 0.0
        else:
            disp = np.array([0.0, 0.0, 0.0])
            angle = 0.0
            speed = 0.0
        return disp, angle, speed


# ============================================================
# Pouring Scene (MPM fluid + rigid container)
# ============================================================

class PouringScene:
    """Multi-physics pouring scene: rigid container + MPM fluid.

    The container is kinematic (scripted motion), fluid is MPM.Liquid.
    """

    def __init__(
        self,
        particle_size: float = 0.015,
        substeps: int = 10,
        dt: float = 5e-4,
        camera: Optional[dict] = None,
        lower_bound: Tuple[float, float, float] = (-1.0, -1.0, 0.0),
        upper_bound: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.particle_size = particle_size
        self.substeps = substeps
        self.dt = dt
        self._camera_cfg = camera
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.scene: Optional[gs.Scene] = None
        self.fluid = None
        self.container = None
        self.cam = None
        self.n_particles = 0
        self.frame_count = 0
        self._step_times = []

    def build(
        self,
        fluid_pos: Tuple[float, float, float] = (-0.2, 0.0, 0.25),
        fluid_size: Tuple[float, float, float] = (0.12, 0.12, 0.2),
        with_scenery: bool = False,
        fluid_color: Optional[Tuple] = None,
    ):
        """Build the pouring scene.

        Args:
            with_scenery: Add shelf platform + receiving container for visual pouring.
        """
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=self.substeps,
                gravity=(0.0, 0.0, -9.81),
            ),
            mpm_options=gs.options.solvers.MPMOptions(
                dt=self.dt,
                particle_size=self.particle_size,
                grid_density=64.0,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
            ),
            rigid_options=gs.options.solvers.RigidOptions(
                dt=self.dt,
                enable_collision=True,
            ),
            show_viewer=False,
        )

        # Ground
        self.scene.add_entity(gs.morphs.Plane())

        # Optional scenery: shelf + receiving container
        if with_scenery:
            shelf_color = (0.55, 0.5, 0.45, 1.0)
            container_color = (0.4, 0.4, 0.45, 1.0)

            # Shelf platform
            self.scene.add_entity(
                gs.morphs.Box(pos=(-0.4, 0.0, 0.2), size=(0.4, 0.5, 0.4), fixed=True),
                surface=gs.surfaces.Default(color=shelf_color),
            )
            # Shelf back wall
            self.scene.add_entity(
                gs.morphs.Box(pos=(-0.62, 0.0, 0.5), size=(0.02, 0.5, 0.3), fixed=True),
                surface=gs.surfaces.Default(color=shelf_color),
            )
            # Shelf side walls
            for y in [-0.25, 0.25]:
                self.scene.add_entity(
                    gs.morphs.Box(pos=(-0.4, y, 0.5), size=(0.4, 0.02, 0.3), fixed=True),
                    surface=gs.surfaces.Default(color=shelf_color),
                )

            # Receiving container
            self.scene.add_entity(
                gs.morphs.Box(pos=(0.45, 0.0, 0.1), size=(0.02, 0.5, 0.2), fixed=True),
                surface=gs.surfaces.Default(color=container_color),
            )
            self.scene.add_entity(
                gs.morphs.Box(pos=(-0.05, 0.0, 0.05), size=(0.02, 0.5, 0.1), fixed=True),
                surface=gs.surfaces.Default(color=container_color),
            )
            for y in [-0.25, 0.25]:
                self.scene.add_entity(
                    gs.morphs.Box(pos=(0.2, y, 0.1), size=(0.5, 0.02, 0.2), fixed=True),
                    surface=gs.surfaces.Default(color=container_color),
                )

        # Fluid block
        fluid_kwargs = dict(
            morph=gs.morphs.Box(pos=fluid_pos, size=fluid_size),
            material=gs.materials.MPM.Liquid(sampler="pbs"),
        )
        if fluid_color:
            fluid_kwargs["surface"] = gs.surfaces.Default(
                color=fluid_color, vis_mode="particle",
            )
        self.fluid = self.scene.add_entity(**fluid_kwargs)

        # Camera (must be added before build)
        if self._camera_cfg:
            self.cam = self.scene.add_camera(**self._camera_cfg)

        self.scene.build()
        self.n_particles = self.fluid.n_particles
        self.frame_count = 0
        self._step_times = []
        return self.n_particles

    def step(self):
        t0 = time.perf_counter()
        self.scene.step()
        elapsed = time.perf_counter() - t0
        self.frame_count += 1
        self._step_times.append(elapsed)

    def set_substeps(self, substeps: int):
        """Change substeps at runtime (fidelity knob)."""
        self.scene.sim_options.substeps = substeps
        self.substeps = substeps

    def get_positions(self) -> np.ndarray:
        return self.fluid.get_state().pos.cpu().numpy().squeeze(0)

    def get_velocities(self) -> np.ndarray:
        return self.fluid.get_state().vel.cpu().numpy().squeeze(0)

    def get_max_speed(self) -> float:
        vel = self.get_velocities()
        return float(np.max(np.linalg.norm(vel, axis=1)))

    def get_kinetic_energy(self) -> float:
        vel = self.get_velocities()
        speed_sq = np.sum(vel ** 2, axis=1)
        vol = self.particle_size ** 3
        mass = 1000.0 * vol
        return float(0.5 * mass * np.sum(speed_sq))

    def get_step_stats(self) -> dict:
        if not self._step_times:
            return {}
        times = np.array(self._step_times)
        return {
            "mean_step_ms": float(times.mean() * 1000),
            "std_step_ms": float(times.std() * 1000),
            "total_frames": self.frame_count,
            "particles": self.n_particles,
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
# Adaptive Fidelity Controller
# ============================================================

class AdaptiveFidelityController:
    """Switches substeps based on detected phase."""

    def __init__(
        self,
        pouring_scene: PouringScene,
        detector: PhaseDetector,
        policy: Optional[SubstepPolicy] = None,
    ):
        self.scene = pouring_scene
        self.detector = detector
        self.policy = policy or SubstepPolicy()

        self.current_phase = SimPhase.STATIC
        self._phase_times: Dict[str, float] = {p.value: 0.0 for p in SimPhase}
        self._phase_steps: Dict[str, int] = {p.value: 0 for p in SimPhase}
        self._switch_log: List[Tuple[int, str, str]] = []
        self._frame = 0

    def step(self, container_speed: float, container_angle: float):
        fluid_max_speed = self.scene.get_max_speed()
        fluid_ke = self.scene.get_kinetic_energy()

        new_phase = self.detector.detect(
            container_speed, container_angle, fluid_max_speed, fluid_ke
        )

        if new_phase != self.current_phase:
            self._switch_log.append(
                (self._frame, self.current_phase.value, new_phase.value)
            )
            new_substeps = self.policy.get(new_phase)
            self.scene.set_substeps(new_substeps)
            self.current_phase = new_phase

        t0 = time.perf_counter()
        self.scene.step()
        elapsed = time.perf_counter() - t0

        self._phase_times[new_phase.value] += elapsed
        self._phase_steps[new_phase.value] += 1
        self._frame += 1

    def get_stats(self) -> dict:
        return {
            "phase_times": dict(self._phase_times),
            "phase_steps": dict(self._phase_steps),
            "total_time": sum(self._phase_times.values()),
            "switches": len(self._switch_log),
            "switch_log": self._switch_log,
        }


# ============================================================
# Comparison benchmark
# ============================================================

def run_comparison(total_frames: int = 400):
    """Compare fixed HIGH substeps vs adaptive substeps."""
    trajectory = ContainerTrajectory(total_frames=total_frames)

    # --- Fixed HIGH ---
    print("Running fixed HIGH substeps (20)...")
    scene_fixed = PouringScene(substeps=20)
    scene_fixed.build()
    detector_fixed = PhaseDetector()

    t0 = time.perf_counter()
    for frame in range(total_frames):
        _, angle, speed = trajectory.get_state(frame)
        fluid_max = scene_fixed.get_max_speed()
        fluid_ke = scene_fixed.get_kinetic_energy()
        detector_fixed.detect(speed, angle, fluid_max, fluid_ke)
        scene_fixed.step()
    fixed_time = time.perf_counter() - t0
    fixed_stats = scene_fixed.get_step_stats()

    # --- Adaptive ---
    print("Running adaptive substeps...")
    scene_adaptive = PouringScene(substeps=10)
    scene_adaptive.build()
    detector_adaptive = PhaseDetector()
    controller = AdaptiveFidelityController(scene_adaptive, detector_adaptive)

    t0 = time.perf_counter()
    for frame in range(total_frames):
        _, angle, speed = trajectory.get_state(frame)
        controller.step(speed, angle)
    adaptive_time = time.perf_counter() - t0
    adaptive_stats = controller.get_stats()

    speedup = fixed_time / adaptive_time if adaptive_time > 0 else 0

    # Position comparison (center of mass)
    fixed_com = scene_fixed.get_positions().mean(axis=0)
    adaptive_com = scene_adaptive.get_positions().mean(axis=0)
    com_error = float(np.linalg.norm(fixed_com - adaptive_com))

    return {
        "fixed_time": fixed_time,
        "adaptive_time": adaptive_time,
        "speedup": speedup,
        "com_error": com_error,
        "fixed_stats": fixed_stats,
        "adaptive_stats": adaptive_stats,
        "fixed_particles": scene_fixed.n_particles,
        "adaptive_particles": scene_adaptive.n_particles,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    gs.init(backend=gs.cuda, logging_level="warning")

    print("=== Genesis Pouring: Fixed vs Adaptive Fidelity ===\n")

    results = run_comparison(total_frames=400)

    print(f"\n{'=' * 55}")
    print(f"{'Metric':<25} {'Fixed HIGH':>14} {'Adaptive':>14}")
    print(f"{'=' * 55}")
    print(f"{'Total time (s)':<25} {results['fixed_time']:>14.2f} {results['adaptive_time']:>14.2f}")
    print(f"{'ms/frame':<25} {results['fixed_time']/400*1000:>14.1f} {results['adaptive_time']/400*1000:>14.1f}")
    print(f"{'Particles':<25} {results['fixed_particles']:>14d} {results['adaptive_particles']:>14d}")
    print(f"{'CoM error (m)':<25} {results['com_error']:>14.6f} {'---':>14}")
    print(f"{'Speedup':<25} {'---':>14} {results['speedup']:>14.2f}x")

    astats = results["adaptive_stats"]
    if "phase_times" in astats:
        print(f"\nAdaptive phase breakdown:")
        for phase in ["transport", "pouring", "settling", "static"]:
            steps = astats["phase_steps"].get(phase, 0)
            t = astats["phase_times"].get(phase, 0)
            print(f"  {phase:<12}: {steps:>4} steps, {t:.3f}s")
        print(f"  Fidelity switches: {astats['switches']}")
