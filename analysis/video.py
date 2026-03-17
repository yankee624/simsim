"""Video rendering for Genesis simulation experiments.

Renders videos using the simulation classes from envs/genesis/.
Each scene uses the same classes used for benchmarking, with camera enabled.

Usage:
    python -m analysis.video --scene dam_break
    python -m analysis.video --scene franka
    python -m analysis.video --scene pouring
    python -m analysis.video --scene cloth
    python -m analysis.video --scene all
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import genesis as gs


# ===========================================================================
# Scene 1: Dam Break (FluidSimulation)
# ===========================================================================

def render_dam_break(
    output_path: str = "videos/dam_break.mp4",
    n_frames: int = 250,
    res: tuple = (1280, 720),
    fps: int = 30,
):
    """Render dam break using FluidSimulation with wall + obstacle."""
    from envs.genesis.fluid_sim import FluidSimulation, FidelityConfig

    print("Rendering: Dam Break...")

    sim = FluidSimulation(
        fidelity=FidelityConfig(
            particle_size=0.015, substeps=10, dt=1e-3, grid_density=64.0,
        ),
        bound_min=(-1.5, -1.5, -0.5),
        bound_max=(2.0, 1.5, 2.0),
        camera=dict(res=res, pos=(1.8, -1.2, 0.9), lookat=(0.0, 0.0, 0.15), fov=50),
    )
    n = sim.build_dam_break(
        fluid_pos=(-0.35, 0.0, 0.3),
        fluid_size=(0.45, 0.6, 0.55),
        walls=True,
        obstacle=True,
        fluid_color=(0.15, 0.45, 0.95, 0.9),
    )
    print(f"  {n} particles")

    sim.start_recording()
    for i in range(n_frames):
        sim.step()
        sim.render_frame()
        if i % 50 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    sim.stop_recording(output_path, fps)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Scene 2: Franka Manipulation (FrankaManipulationEnv)
# ===========================================================================

def render_franka(
    output_path: str = "videos/franka.mp4",
    n_frames: int = 300,
    res: tuple = (1280, 720),
    fps: int = 30,
):
    """Render Franka arm with sinusoidal joint motions near a box."""
    from envs.genesis.manipulation import FrankaManipulationEnv, RigidFidelityConfig

    print("Rendering: Franka Manipulation...")

    env = FrankaManipulationEnv(
        fidelity=RigidFidelityConfig(
            dt=5e-3, substeps=8, solver_iterations=50, ls_iterations=50,
        ),
        camera=dict(res=res, pos=(1.3, -1.0, 1.0), lookat=(0.3, 0.0, 0.35), fov=50),
        box_color=(0.9, 0.25, 0.2, 1.0),
        show_target_marker=True,
    )
    env.build()

    env.start_recording()
    for i in range(n_frames):
        t = i / n_frames
        action = np.array([
            5.0 * np.sin(2 * np.pi * t),
            3.0 * np.cos(2 * np.pi * t),
            2.0 * np.sin(4 * np.pi * t),
            -4.0 * np.cos(2 * np.pi * t),
            1.0 * np.sin(3 * np.pi * t),
            3.0 * np.cos(3 * np.pi * t),
            1.0 * np.sin(5 * np.pi * t),
        ])
        env.step(action)
        env.render_frame()

        if i % 75 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    env.stop_recording(output_path, fps)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Scene 3: Pouring (PouringScene with shelf scenery)
# ===========================================================================

def render_pouring(
    output_path: str = "videos/pouring.mp4",
    n_frames: int = 300,
    res: tuple = (1280, 720),
    fps: int = 30,
):
    """Render fluid pouring off a shelf into a receiving container."""
    from envs.genesis.pouring import PouringScene

    print("Rendering: Pouring...")

    scene = PouringScene(
        particle_size=0.015,
        substeps=20,
        dt=4e-3,
        camera=dict(res=res, pos=(1.5, -1.5, 1.0), lookat=(0.0, 0.0, 0.25), fov=50),
        lower_bound=(-1.5, -1.5, -0.5),
        upper_bound=(1.5, 1.5, 2.0),
    )
    n = scene.build(
        fluid_pos=(-0.35, 0.0, 0.55),
        fluid_size=(0.25, 0.35, 0.2),
        with_scenery=True,
        fluid_color=(0.15, 0.45, 0.95, 0.9),
    )
    print(f"  {n} particles")

    scene.start_recording()
    for i in range(n_frames):
        scene.step()
        scene.render_frame()
        if i % 75 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    scene.stop_recording(output_path, fps)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Scene 4: Cup Carrying (CupCarryingSim)
# ===========================================================================

def render_cup_carry(
    output_path: str = "videos/cup_carry.mp4",
    n_frames: int = 200,
    res: tuple = (1280, 720),
    fps: int = 30,
):
    """Render cup carrying: liquid-filled cup transported along figure-8 path."""
    from envs.genesis.cup_carrying import CupCarryingSim, CARRY_MEDIUM, CarryTrajectory

    print("Rendering: Cup Carrying...")

    sim = CupCarryingSim(
        fidelity=CARRY_MEDIUM,
        cup_init_pos=(0.0, 0.0, 0.0),
        camera=dict(res=res, pos=(0.5, -0.5, 0.45), lookat=(0.0, 0.0, 0.1), fov=55),
        bound_min=(-0.8, -0.8, -0.1),
        bound_max=(0.8, 0.8, 0.8),
    )
    n = sim.build(fluid_color=(0.15, 0.45, 0.95, 0.9))
    print(f"  {n} particles")

    traj = CarryTrajectory.figure_eight(n_frames, radius=0.08)

    sim.start_recording()
    for i, delta in enumerate(traj.deltas):
        sim.step(delta)
        sim.render_frame()
        if i % 50 == 0:
            spill = sim.get_spill_fraction()
            print(f"  Frame {i+1}/{n_frames}  spill={spill:.4f}")

    sim.stop_recording(output_path, fps)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Scene 5: Cloth Drop (PBD ClothSimulation)
# ===========================================================================

def render_cloth(
    output_path: str = "videos/cloth.mp4",
    n_frames: int = 150,
    res: tuple = (1280, 720),
    fps: int = 30,
):
    """Render IPC cloth dropping and draping over a rigid sphere."""
    # Preload system CUDA 12.9 cublas — required for libuipc IPC coupler only.
    import ctypes
    _cublas = "/usr/local/cuda-12.9/lib64/libcublas.so.12"
    if os.path.exists(_cublas):
        ctypes.CDLL(_cublas, mode=ctypes.RTLD_GLOBAL)

    from envs.genesis.cloth import ClothSimulation, ClothFidelityConfig

    print("Rendering: IPC Cloth Drop...")

    sim = ClothSimulation(
        fidelity=ClothFidelityConfig(
            mesh_resolution=30, substeps=10, dt=2e-3,
        ),
        camera=dict(res=res, pos=(0.7, -0.7, 0.5), lookat=(0.0, 0.0, 0.2), fov=50),
    )
    n = sim.build_cloth_drop(
        cloth_size=(0.5, 0.5),
        cloth_center=(0.0, 0.0, 0.5),
        obstacle=True,
        cloth_color=(0.2, 0.6, 0.9, 1.0),
    )
    print(f"  {n} particles")

    sim.start_recording()
    for i in range(n_frames):
        sim.step()
        sim.render_frame()
        if i % 30 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    sim.stop_recording(output_path, fps)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

SCENES = {
    "dam_break": render_dam_break,
    "franka": render_franka,
    "pouring": render_pouring,
    "cup_carry": render_cup_carry,
    "cloth": render_cloth,
}


def main():
    parser = argparse.ArgumentParser(description="Render Genesis simulation videos")
    parser.add_argument(
        "--scene", choices=list(SCENES.keys()) + ["all"],
        default="all", help="Scene to render",
    )
    parser.add_argument("--output-dir", default="videos", help="Output directory")
    parser.add_argument(
        "--res", default="1280x720",
        help="Resolution WxH (default: 1280x720)",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    args = parser.parse_args()

    import os
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.scene == "all":
        # Run each scene in a separate process to avoid EGL context conflicts
        for name in SCENES:
            print(f"--- Launching {name} in subprocess ---")
            t0 = time.time()
            cmd = [sys.executable, "-m", "analysis.video",
                   "--scene", name,
                   "--output-dir", args.output_dir,
                   "--res", args.res]
            if args.gpu is not None:
                cmd.extend(["--gpu", str(args.gpu)])
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  ERROR rendering {name} (exit code {result.returncode})")
            else:
                print(f"  Completed in {time.time() - t0:.1f}s\n")
        print(f"Done! Videos saved in {args.output_dir}/")
    else:
        gs.init(backend=gs.cuda, logging_level="warning")

        w, h = map(int, args.res.split("x"))
        res = (w, h)

        render_fn = SCENES[args.scene]
        output_path = f"{args.output_dir}/{args.scene}.mp4"
        t0 = time.time()
        render_fn(output_path=output_path, res=res)
        print(f"  Completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
