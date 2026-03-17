"""Genesis IPC-based cloth simulation with fidelity controls.

Uses FEM.Cloth (thin-shell) with IPC coupler for physically accurate
collision handling. The IPC (Incremental Potential Contact) method
provides guaranteed no-penetration via barrier energy.

Requires: libuipc built with CUDA sm_86 (RTX 3090).
Preload system CUDA 12.9 cublas before importing genesis.

Fidelity knobs:
  - mesh_resolution: vertices per side of cloth grid (higher = finer mesh)
  - substeps: physics substeps per scene step
  - dt: simulation timestep
  - E: Young's modulus (stiffness)
  - thickness: shell thickness for IPC contact
"""

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Preload system CUDA 12.9 cublas (conda's is too old for libuipc)
if os.path.exists("/usr/local/cuda-12.9/lib64/libcublas.so.12"):
    ctypes.CDLL("/usr/local/cuda-12.9/lib64/libcublas.so.12", mode=ctypes.RTLD_GLOBAL)

import genesis as gs


# ============================================================
# Fidelity Configuration
# ============================================================

@dataclass
class ClothFidelityConfig:
    """Tunable fidelity parameters for IPC cloth simulation."""
    mesh_resolution: int = 20      # vertices per side (NxN grid)
    substeps: int = 10
    dt: float = 4e-3

    # Material
    rho: float = 400.0             # density (kg/m^3)
    E: float = 1e5                 # Young's modulus (Pa)
    nu: float = 0.3                # Poisson's ratio
    thickness: float = 0.0005      # shell thickness (m) — must be << vertex spacing

    @property
    def label(self) -> str:
        return (f"res={self.mesh_resolution},sub={self.substeps},"
                f"dt={self.dt},E={self.E:.0e}")


CLOTH_FIDELITY_LOW = ClothFidelityConfig(
    mesh_resolution=10, substeps=5, dt=5e-3, E=5e4,
)
CLOTH_FIDELITY_MEDIUM = ClothFidelityConfig(
    mesh_resolution=20, substeps=10, dt=4e-3, E=1e5,
)
CLOTH_FIDELITY_HIGH = ClothFidelityConfig(
    mesh_resolution=30, substeps=15, dt=3e-3, E=2e5,
)


# ============================================================
# Mesh Generation
# ============================================================

def generate_cloth_obj(
    filepath: str,
    width: float = 0.5,
    height: float = 0.5,
    nx: int = 20,
    ny: int = 20,
) -> str:
    """Generate a flat rectangular cloth surface mesh as an OBJ file.

    The mesh is centered at the origin in the XY plane (z=0).
    Position offset is applied via gs.morphs.Mesh(pos=...).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = []
    for j in range(ny):
        for i in range(nx):
            x = (i / (nx - 1) - 0.5) * width
            y = (j / (ny - 1) - 0.5) * height
            vertices.append((x, y, 0.0))

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j * nx + i
            # Two triangles per quad (OBJ is 1-indexed)
            faces.append((v0 + 1, v0 + 2, v0 + nx + 2))
            faces.append((v0 + 1, v0 + nx + 2, v0 + nx + 1))

    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    return str(path)


# ============================================================
# Cloth Simulation
# ============================================================

class ClothSimulation:
    """Genesis IPC cloth simulation with rigid body interaction.

    Uses FEM.Cloth material with IPC coupler for physically accurate
    collision handling (guaranteed no-penetration).

    Usage:
        sim = ClothSimulation(fidelity=CLOTH_FIDELITY_MEDIUM)
        sim.build_cloth_drop(obstacle=True)
        for _ in range(200):
            sim.step()
    """

    def __init__(
        self,
        fidelity: Optional[ClothFidelityConfig] = None,
        camera: Optional[dict] = None,
    ):
        self.fidelity = fidelity or CLOTH_FIDELITY_MEDIUM
        self._camera_cfg = camera

        self.scene: Optional[gs.Scene] = None
        self.cloth_entity = None
        self.cam = None
        self.n_particles = 0
        self.frame_count = 0
        self._step_times = []

    def build_cloth_drop(
        self,
        cloth_size: Tuple[float, float] = (0.5, 0.5),
        cloth_center: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        obstacle: bool = True,
        obstacle_pos: Tuple[float, float, float] = (0.0, 0.0, 0.25),
        obstacle_radius: float = 0.12,
        cloth_color: Optional[Tuple] = None,
    ):
        """Build a cloth drop scene: cloth falls and drapes over an obstacle.

        Uses IPC coupler for physically accurate collision.
        """
        f = self.fidelity

        mesh_path = generate_cloth_obj(
            filepath="/tmp/genesis_ipc_cloth.obj",
            width=cloth_size[0],
            height=cloth_size[1],
            nx=f.mesh_resolution,
            ny=f.mesh_resolution,
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=f.dt,
                substeps=f.substeps,
                gravity=(0.0, 0.0, -9.81),
            ),
            coupler_options=gs.options.IPCCouplerOptions(
                contact_enable=True,
                contact_friction_enable=True,
                sanity_check_enable=False,
            ),
            fem_options=gs.options.solvers.FEMOptions(dt=f.dt),
            rigid_options=gs.options.solvers.RigidOptions(
                dt=f.dt, enable_collision=True,
            ),
            show_viewer=False,
        )

        # Ground plane (IPC-coupled)
        self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(coup_type="ipc_only"),
        )

        # Obstacle (IPC-coupled)
        if obstacle:
            self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=obstacle_pos,
                    radius=obstacle_radius,
                    fixed=True,
                ),
                material=gs.materials.Rigid(coup_type="ipc_only"),
                surface=gs.surfaces.Default(color=(0.8, 0.3, 0.2, 1.0)),
            )

        # Cloth (FEM.Cloth thin-shell + IPC contact)
        cloth_kwargs = dict(
            morph=gs.morphs.Mesh(file=mesh_path, pos=cloth_center),
            material=gs.materials.FEM.Cloth(
                rho=f.rho,
                E=f.E,
                nu=f.nu,
                thickness=f.thickness,
            ),
        )
        if cloth_color:
            cloth_kwargs["surface"] = gs.surfaces.Default(color=cloth_color)
        self.cloth_entity = self.scene.add_entity(**cloth_kwargs)

        # Camera
        if self._camera_cfg:
            self.cam = self.scene.add_camera(**self._camera_cfg)

        self.scene.build()

        self.n_particles = self.cloth_entity.n_vertices
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
        """Get cloth particle positions as (N, 3) numpy array."""
        pos = self.cloth_entity.get_state().pos
        if pos.ndim == 3:
            pos = pos.squeeze(0)
        return pos.cpu().numpy()

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
    import argparse
    parser = argparse.ArgumentParser(description="IPC Cloth Benchmark")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    gs.init(backend=gs.cuda, logging_level="warning")

    print("=== Genesis IPC Cloth Simulation Benchmark ===\n")

    for name, fidelity in [("LOW", CLOTH_FIDELITY_LOW), ("MEDIUM", CLOTH_FIDELITY_MEDIUM), ("HIGH", CLOTH_FIDELITY_HIGH)]:
        sim = ClothSimulation(fidelity=fidelity)
        n = sim.build_cloth_drop()
        print(f"[{name}] {fidelity.label}")
        print(f"  Particles: {n}")

        # Warm-up
        sim.step()

        # Benchmark
        n_frames = 20
        t0 = time.perf_counter()
        for _ in range(n_frames):
            sim.step()
        elapsed = time.perf_counter() - t0

        stats = sim.get_step_stats()
        print(f"  {n_frames} frames in {elapsed:.2f}s ({elapsed/n_frames*1000:.1f} ms/frame)")
        print(f"  Stats: {stats['mean_step_ms']:.1f} +/- {stats['std_step_ms']:.1f} ms/frame")
        print()
