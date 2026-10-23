import os
import csv
from typing import Optional
import numpy as np
import pychrono as chrono


class ExcavationTelemetry:
    """
    Tracks forces, torques, drum kinematics, and SPH particle mass fractions
    (collected inside inner cavity vs. carried in outer scoop vs. spilled).
    """

    def __init__(
        self,
        env,
        rig,
        output_dir: str,
        save_particles: bool = True,
        prefix: Optional[str] = None,
    ):
        self.env = env
        self.rig = rig
        self.output_dir = output_dir
        self.save_particles = save_particles
        self.prefix = prefix

        self.particles_dir = os.path.join(output_dir, "particles")
        os.makedirs(self.particles_dir, exist_ok=True)

        # Mass of a single SPH continuum particle: m_p = rho_0 * d_0^3
        rho0 = float(env.cfg["soil_material"]["bulk_density"])
        d0 = float(env.cfg["sph_discretization"]["initial_spacing"])
        self.particle_mass_kg = rho0 * (d0 ** 3)

        # Drum geometric bounds in local drum coordinates (centered at origin, axis along Y)
        self.r_inner = rig.inner_radius
        self.r_outer = rig.outer_radius
        self.half_len = 0.5 * rig.drum_length

        # State tracking for collection and spill rate derivatives
        self.prev_time = 0.0
        self.prev_collected_mass_kg = 0.0
        self.cumulative_spilled_mass_kg = 0.0
        self.engaged_particle_ids = set()  # Particles lifted above Z=0 inside outer radius

        # Keep standard filenames so replay_irrlicht.py works seamlessly
        self.metrics_path = os.path.join(output_dir, "metrics_timeseries.csv")
        self.traj_path = os.path.join(output_dir, "drum_trajectory.csv")

        self._f_met = open(self.metrics_path, "w", newline="", encoding="utf-8")
        self._f_traj = open(self.traj_path, "w", newline="", encoding="utf-8")

        self.met_writer = csv.writer(self._f_met)
        self.traj_writer = csv.writer(self._f_traj)

        self.met_writer.writerow([
            "frame", "time_s",
            "Fx_N", "Fy_N", "Fz_N",
            "Tx_Nm", "Ty_Nm", "Tz_Nm", "motor_torque_Nm",
            "collected_mass_kg", "in_scoop_mass_kg", "cumulative_spilled_mass_kg",
            "collection_rate_kg_s", "spill_rate_kg_s"
        ])
        self.traj_writer.writerow([
            "frame", "time_s", "px", "py", "pz", "e0", "e1", "e2", "e3"
        ])

    def _extract_sph_positions(self) -> np.ndarray:
        """Retrieves current SPH fluid/soil particle 3D positions from ChFsiFluidSystemSPH."""
        pos_vec = self.env.sysSPH.GetParticlePositions()
        n_part = len(pos_vec)
        if n_part == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Fast extraction supporting PyChrono ChVector3d lists
        coords = np.array([[p.x, p.y, p.z] for p in pos_vec], dtype=np.float32)
        return coords

    def record_frame(
        self,
        frame_idx: int,
        sim_time: float,
        pts_world: Optional[np.ndarray] = None,
    ):
        drum = self.rig.drum_body
        pos = drum.GetPos()
        rot = drum.GetRot()

        # 1. Record 6-DOF Drum Trajectory for Irrlicht Replay
        self.traj_writer.writerow([
            frame_idx, f"{sim_time:.5f}",
            f"{pos.x:.6f}", f"{pos.y:.6f}", f"{pos.z:.6f}",
            f"{rot.e0:.6f}", f"{rot.e1:.6f}", f"{rot.e2:.6f}", f"{rot.e3:.6f}"
        ])

        # 2. Extract FSI Forces & Torques Acting on the Bucket Drum
        if self.rig.fsi_body_index is not None:
            f_fsi = self.env.sysFSI.GetFsiBodyForce(self.rig.fsi_body_index)
            t_fsi = self.env.sysFSI.GetFsiBodyTorque(self.rig.fsi_body_index)
        else:
            f_fsi = drum.GetAppliedForce()
            t_fsi = drum.GetAppliedTorque()

        motor_trq = self.rig.rot_motor.GetMotorTorque()

        # 3. Classify SPH Particles (Collected vs. In-Scoop vs. Spilled)
        if pts_world is None:
            pts_world = self._extract_sph_positions()

        collected_mass = 0.0
        in_scoop_mass = 0.0
        step_spilled_mass = 0.0

        if pts_world.shape[0] > 0:
            # Vectorized world-to-local transformation using drum quaternion rotation matrix
            r_mat = chrono.ChMatrix33d(rot)
            ax = r_mat.GetAxisX()
            ay = r_mat.GetAxisY()
            az = r_mat.GetAxisZ()

            R = np.array([
                [ax.x, ay.x, az.x],
                [ax.y, ay.y, az.y],
                [ax.z, ay.z, az.z],
            ], dtype=np.float32)

            t_vec = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

            # p_local = R^T * (p_world - t_vec)
            pts_local = (pts_world - t_vec) @ R
            radial_dist = np.hypot(pts_local[:, 0], pts_local[:, 2])
            within_length = np.abs(pts_local[:, 1]) <= self.half_len

            # Mask A: Collected inside internal baffle cavity (r <= r_inner)
            mask_collected = within_length & (radial_dist <= self.r_inner)
            # Mask B: In outer scoop channel (r_inner < r <= r_outer)
            mask_scoop = within_length & (radial_dist > self.r_inner) & (radial_dist <= self.r_outer)

            collected_mass = float(np.count_nonzero(mask_collected)) * self.particle_mass_kg
            in_scoop_mass = float(np.count_nonzero(mask_scoop)) * self.particle_mass_kg

            # Spill tracking: particles lifted above bed surface (Z > 0.005m) inside drum envelope
            lifted_inside_ids = set(
                np.where((mask_collected | mask_scoop) & (pts_world[:, 2] > 0.005))[0]
            )
            outside_envelope_ids = set(
                np.where((~within_length) | (radial_dist > self.r_outer))[0]
            )

            # Any particle previously lifted inside the drum that falls back outside is counted as spilled
            newly_spilled_ids = self.engaged_particle_ids.intersection(outside_envelope_ids)
            step_spilled_mass = float(len(newly_spilled_ids)) * self.particle_mass_kg
            self.cumulative_spilled_mass_kg += step_spilled_mass

            # Update engaged set
            self.engaged_particle_ids = (self.engaged_particle_ids - newly_spilled_ids) | lifted_inside_ids

            # Save compressed NumPy particle state for fast Irrlicht playback
            if self.save_particles:
                state_code = np.zeros(pts_world.shape[0], dtype=np.uint8)
                state_code[mask_scoop] = 1
                state_code[mask_collected] = 2
                np.savez_compressed(
                    os.path.join(self.particles_dir, f"frame_{frame_idx:04d}.npz"),
                    pos=pts_world,
                    state=state_code,
                )

        # Compute instantaneous rates
        dt_out = max(sim_time - self.prev_time, 1e-6)
        collection_rate = (collected_mass - self.prev_collected_mass_kg) / dt_out
        spill_rate = step_spilled_mass / dt_out

        self.prev_time = sim_time
        self.prev_collected_mass_kg = collected_mass

        self.met_writer.writerow([
            frame_idx, f"{sim_time:.5f}",
            f"{f_fsi.x:.4f}", f"{f_fsi.y:.4f}", f"{f_fsi.z:.4f}",
            f"{t_fsi.x:.4f}", f"{t_fsi.y:.4f}", f"{t_fsi.z:.4f}", f"{motor_trq:.4f}",
            f"{collected_mass:.5f}", f"{in_scoop_mass:.5f}", f"{self.cumulative_spilled_mass_kg:.5f}",
            f"{collection_rate:.5f}", f"{spill_rate:.5f}"
        ])
        self._f_met.flush()
        self._f_traj.flush()

    def close(self):
        self._f_met.close()
        self._f_traj.close()
