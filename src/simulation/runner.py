import os
import sys
import time
import argparse

# ---------------------------------------------------------------------------
# PyCharm Direct Execution Fix:
# Automatically add project root to sys.path and set working directory to root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(str(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

try:
    from .environment import CRMExcavationEnvironment, load_yaml_config
    from .drum_rig import BucketDrumRig
    from .telemetry import ExcavationTelemetry
except ImportError:
    try:
        from simulation.environment import CRMExcavationEnvironment, load_yaml_config
        from simulation.drum_rig import BucketDrumRig
        from simulation.telemetry import ExcavationTelemetry
    except ImportError:
        from simulation.environment import CRMExcavationEnvironment, load_yaml_config
        from simulation.drum_rig import BucketDrumRig
        from simulation.telemetry import ExcavationTelemetry


def run_sequential_simulation(
    crm_cfg: dict,
    drum_cfg: dict,
    queue_item: dict,
    output_root: str,
    save_particles: bool = True,
    verbose: bool = False,
):
    """Executes a single isolated GPU-accelerated simulation run from the queue."""
    run_id = queue_item["run_id"]
    output_dir = os.path.join(output_root, run_id)
    os.makedirs(output_dir, exist_ok=True)

    op_cfg = {
        "rpm": queue_item["kinematics"]["rpm"],
        "forward_velocity": queue_item["kinematics"]["forward_velocity"],
        "plunge_duration": queue_item["kinematics"]["plunge_duration"],
        "cut_depth": queue_item["path"]["cut_depth"],
        "start_x": queue_item["path"]["start_x"],
    }

    y_offset = float(queue_item["path"].get("y_offset", 0.0))

    print(f"\n=======================================================")
    print(f"Executing Queue Run: {run_id}")
    print(f"  PLA Drum: {drum_cfg['id']} (L={drum_cfg['drum_length']}m, R={drum_cfg['outer_radius']}m)")
    print(f"  Path: X_start={op_cfg['start_x']}m, Y_offset={y_offset}m, Cut={op_cfg['cut_depth']}m")
    print(f"  Kinematics: {op_cfg['rpm']} RPM, Vx={op_cfg['forward_velocity']} m/s")
    print(f"=======================================================")

    # 1. Instantiate fresh physical SMC and CUDA CRM systems
    env = CRMExcavationEnvironment(crm_cfg, verbose=verbose)

    # 2. Build PLA drum rig and sample surface BCE markers
    rig = BucketDrumRig(env, drum_cfg, op_cfg, y_offset=y_offset)

    # 3. Fill container with BP-1 soil and transfer buffers to CUDA GPU
    env.build_regolith_pit()

    # 4. Initialize telemetry
    telemetry = ExcavationTelemetry(
        env, rig, output_dir, save_particles=save_particles, prefix=drum_cfg["id"]
    )

    # 5. GPU Time-Stepping Loop
    dt_mbd = float(crm_cfg["simulation"]["dt_mbd"])
    duration = float(crm_cfg["simulation"]["duration"])
    out_fps = float(crm_cfg["simulation"]["output_fps"])

    sim_time = 0.0
    frame_idx = 0
    wall_start = time.time()

    try:
        while sim_time < duration:
            # Update native C++ motor speeds only when crossing phase boundaries
            rig.update_motor_phase(sim_time)

            # Advance CUDA SPH + MBS solver
            env.step(dt_mbd)
            sim_time += dt_mbd

            if int(sim_time * out_fps) > frame_idx:
                frame_idx += 1
                telemetry.record_frame(frame_idx, sim_time)

                # Print status every 10 frames (~0.16s of sim time) so you can watch progress live
                if frame_idx % 10 == 0:
                    elapsed = time.time() - wall_start
                    idx = int(rig.fsi_body_index) if isinstance(rig.fsi_body_index, int) else 0
                    f_fsi = env.sysFSI.GetFsiBodyForce(idx)
                    print(
                        f"  [t = {sim_time:5.2f} / {duration:.2f} s] "
                        f"Fx: {f_fsi.x:6.1f} N | Fz: {f_fsi.z:6.1f} N | "
                        f"Collected: {telemetry.prev_collected_mass_kg * 1000:6.1f} g | "
                        f"Spilled: {telemetry.cumulative_spilled_mass_kg * 1000:6.1f} g | "
                        f"Wall: {elapsed:6.1f} s"
                    )
    finally:
        telemetry.close()

    print(f"Finished Run {run_id}. Data saved in: {output_dir}")


def execute_queue(crm_yaml_path: str, sweep_yaml_path: str, verbose: bool = False):
    """Loads config files and loops through the execution queue step-by-step."""
    crm_cfg = load_yaml_config(crm_yaml_path)
    sweep_cfg = load_yaml_config(sweep_yaml_path)

    output_root = sweep_cfg["batch_settings"]["output_root"]
    save_particles = bool(sweep_cfg["batch_settings"].get("save_particle_clouds", True))

    drum_pool = {d["id"]: d for d in sweep_cfg["drums"]}
    queue = sweep_cfg["run_queue"]

    total_runs = len(queue)
    print(f"Working Directory: {os.getcwd()}")
    print(f"Loaded run queue. Total sequential runs to process: {total_runs}")

    for idx, queue_item in enumerate(queue, start=1):
        drum_id = queue_item["drum_id"]
        if drum_id not in drum_pool:
            print(f"Error: Drum ID '{drum_id}' specified in run '{queue_item['run_id']}' is not defined.")
            continue

        drum_cfg = drum_pool[drum_id]
        print(f"\n--- Queue Progress: [{idx}/{total_runs}] ---")

        run_sequential_simulation(
            crm_cfg=crm_cfg,
            drum_cfg=drum_cfg,
            queue_item=queue_item,
            output_root=output_root,
            save_particles=save_particles,
            verbose=verbose,
        )


if __name__ == "__main__":
    DEFAULT_CRM_CONFIG = "configs/bp1_earth_crm.yaml"
    DEFAULT_SWEEP_CONFIG = "configs/sweep_matrix.yaml"
    DEFAULT_VERBOSE = False

    parser = argparse.ArgumentParser(description="Process Sequential CRM Multi-Drum Execution Queue")
    parser.add_argument("--crm-config", default=DEFAULT_CRM_CONFIG, help="Path to CRM YAML")
    parser.add_argument("--sweep-config", default=DEFAULT_SWEEP_CONFIG, help="Path to Sweep YAML")
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Enable verbose logging")
    args = parser.parse_args()

    execute_queue(args.crm_config, args.sweep_config, verbose=args.verbose)
