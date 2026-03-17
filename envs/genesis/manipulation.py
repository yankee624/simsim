"""Genesis-based Franka Panda manipulation environment.

Franka 7-DOF arm performing pick-and-place / pushing tasks.
Supports batched simulation via Genesis scene parallelism.

Fidelity knobs:
  - dt / substeps: temporal resolution
  - constraint solver iterations: contact accuracy
  - collision detection settings
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import genesis as gs


@dataclass
class RigidFidelityConfig:
    """Fidelity parameters for rigid-body simulation."""
    dt: float = 0.005
    substeps: int = 10
    solver_iterations: int = 50
    ls_iterations: int = 50

    @property
    def label(self) -> str:
        return f"dt={self.dt},sub={self.substeps},iter={self.solver_iterations}"


RIGID_FIDELITY_LOW = RigidFidelityConfig(dt=0.01, substeps=4, solver_iterations=20, ls_iterations=20)
RIGID_FIDELITY_MEDIUM = RigidFidelityConfig(dt=0.005, substeps=10, solver_iterations=50, ls_iterations=50)
RIGID_FIDELITY_HIGH = RigidFidelityConfig(dt=0.002, substeps=20, solver_iterations=100, ls_iterations=100)


class FrankaManipulationEnv:
    """Franka Panda manipulation environment using Genesis.

    Task: Push/move a box to a target position on a table.
    Observation: joint angles (7), joint velocities (7), box pos (3), target pos (3)
    Action: joint velocity commands (7)

    Usage:
        env = FrankaManipulationEnv()
        env.build()
        obs = env.get_obs()
        for _ in range(100):
            action = np.random.randn(7) * 0.1
            env.step(action)
    """

    def __init__(
        self,
        fidelity: Optional[RigidFidelityConfig] = None,
        target_pos: Tuple[float, float, float] = (0.5, 0.0, 0.05),
        camera: Optional[dict] = None,
        box_color: Optional[Tuple] = None,
        show_target_marker: bool = False,
    ):
        self.fidelity = fidelity or RIGID_FIDELITY_MEDIUM
        self.target_pos = np.array(target_pos)
        self._camera_cfg = camera
        self._box_color = box_color
        self._show_target_marker = show_target_marker

        self.scene: Optional[gs.Scene] = None
        self.franka = None
        self.box = None
        self.cam = None
        self.frame_count = 0
        self._step_times = []

    def build(self):
        """Build the manipulation scene."""
        f = self.fidelity

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=f.dt,
                substeps=f.substeps,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.solvers.RigidOptions(
                dt=f.dt,
                enable_collision=True,
                iterations=f.solver_iterations,
                ls_iterations=f.ls_iterations,
            ),
            show_viewer=False,
        )

        # Ground
        self.scene.add_entity(gs.morphs.Plane())

        # Franka Panda (fixed base)
        self.franka = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/panda_bullet/panda.urdf",
                pos=(0.0, 0.0, 0.0),
                fixed=True,
            ),
        )

        # Box to manipulate
        box_kwargs = dict(
            morph=gs.morphs.Box(pos=(0.45, 0.0, 0.025), size=(0.05, 0.05, 0.05), fixed=False),
            material=gs.materials.Rigid(friction=0.8),
        )
        if self._box_color:
            box_kwargs["surface"] = gs.surfaces.Default(color=self._box_color)
        self.box = self.scene.add_entity(**box_kwargs)

        # Target marker
        if self._show_target_marker:
            self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=tuple(self.target_pos), radius=0.015, fixed=True,
                ),
                surface=gs.surfaces.Default(color=(0.1, 0.9, 0.2, 0.7)),
            )

        # Camera (must be added before build)
        if self._camera_cfg:
            self.cam = self.scene.add_camera(**self._camera_cfg)

        self.scene.build()

        # Set initial joint positions (ready pose)
        init_qpos = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.785, 0.04, 0.04])
        self.franka.set_qpos(init_qpos)

        self.frame_count = 0
        self._step_times = []

    def step(self, action: np.ndarray):
        """Step with joint velocity action (7 or 9 DOFs)."""
        dof_vel = np.zeros(self.franka.n_dofs)
        n = min(len(action), self.franka.n_dofs)
        dof_vel[:n] = action[:n]
        self.franka.control_dofs_velocity(dof_vel)

        t0 = time.perf_counter()
        self.scene.step()
        elapsed = time.perf_counter() - t0

        self.frame_count += 1
        self._step_times.append(elapsed)

    def get_obs(self) -> dict:
        """Get observation dictionary."""
        qpos = self.franka.get_dofs_position().cpu().numpy()
        qvel = self.franka.get_dofs_velocity().cpu().numpy()

        box_pos = self.box.get_pos().cpu().numpy().flatten()

        return {
            "joint_pos": qpos[:7],
            "joint_vel": qvel[:7],
            "box_pos": box_pos[:3],
            "target_pos": self.target_pos,
        }

    def compute_reward(self) -> float:
        """Reward = negative L2 distance from box to target."""
        obs = self.get_obs()
        return -float(np.linalg.norm(obs["box_pos"][:2] - self.target_pos[:2]))

    def get_step_stats(self) -> dict:
        if not self._step_times:
            return {}
        times = np.array(self._step_times)
        return {
            "mean_step_ms": float(times.mean() * 1000),
            "std_step_ms": float(times.std() * 1000),
            "total_frames": self.frame_count,
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
# CLI
# ============================================================

if __name__ == "__main__":
    gs.init(backend=gs.cuda, logging_level="warning")

    print("=== Genesis Franka Manipulation Benchmark ===\n")

    for name, fidelity in [("LOW", RIGID_FIDELITY_LOW), ("MEDIUM", RIGID_FIDELITY_MEDIUM), ("HIGH", RIGID_FIDELITY_HIGH)]:
        env = FrankaManipulationEnv(fidelity=fidelity)
        env.build()

        obs = env.get_obs()
        print(f"[{name}] {fidelity.label}")
        print(f"  Franka DOFs: {env.franka.n_dofs}")
        print(f"  Initial box pos: {obs['box_pos']}")
        print(f"  Target pos: {obs['target_pos']}")

        # Warm-up
        env.step(np.zeros(7))

        # Benchmark
        n_steps = 100
        t0 = time.perf_counter()
        for i in range(n_steps):
            action = np.random.randn(7) * 0.3
            env.step(action)
        elapsed = time.perf_counter() - t0

        stats = env.get_step_stats()
        reward = env.compute_reward()
        obs = env.get_obs()

        print(f"  {n_steps} steps in {elapsed:.2f}s ({elapsed/n_steps*1000:.1f} ms/step)")
        print(f"  Final box pos: {obs['box_pos']}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Stats: {stats['mean_step_ms']:.1f} +/- {stats['std_step_ms']:.1f} ms/step")
        print()
