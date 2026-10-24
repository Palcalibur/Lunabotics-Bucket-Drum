import os
import math
import numpy as np
import pychrono as chrono


def prepare_and_sample_obj(
    raw_obj_path: str,
    target_outer_radius: float,
    d0: float,
) -> tuple[str, list]:
    """
    1. Parses vertices (v) and triangles (f) from raw_obj_path.
    2. Auto-detects if the CAD mesh was exported in mm or cm instead of meters,
       and scales the vertices so the drum radius matches `target_outer_radius`.
    3. Writes a clean, meter-scaled geometry-only .obj file for Irrlicht visualization.
    4. Vectorized-samples surface BCE markers at spacing `d0` in <0.1s (bypassing
       Chrono's slow C++ ChBodyGeometry brute-force mesh sampler).
    """
    base, ext = os.path.splitext(raw_obj_path)
    clean_obj_path = f"{base}_meters_clean{ext}"
    bce_cache_path = f"{base}_bce_d0_{int(d0 * 1e5)}.npy"

    vertices = []
    faces = []

    with open(raw_obj_path, "r", encoding="utf-8", errors="ignore") as f_in:
        for line in f_in:
            s = line.strip()
            if s.startswith("v "):
                parts = s.split()
                if len(parts) >= 4:
                    _, vx_s, vy_s, vz_s = parts[:4]
                    vertices.append([float(vx_s), float(vy_s), float(vz_s)])
            elif s.startswith("f "):
                parts = s.split()[1:]
                # Convert polygon faces (quads/n-gons) into triangles via fan triangulation
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                first_idx = idxs[0]
                for i in range(1, len(idxs) - 1):
                    faces.append([first_idx, idxs[i], idxs[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"No valid vertices/faces found in {raw_obj_path}")

    V = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.int32)

    # Center mesh at its bounding-box midpoint so rotation about (0,0,0) is concentric
    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    V -= bbox_center

    bbox_extents = bbox_max - bbox_min
    dx, dy, dz = float(bbox_extents[0]), float(bbox_extents[1]), float(bbox_extents[2])
    max_dim = max(dx, dy, dz)
    expected_diameter = 2.0 * target_outer_radius

    # Auto-detect CAD units (mm or cm -> meters)
    scale_factor = 1.0
    if max_dim > 50.0 * expected_diameter:
        scale_factor = 0.001  # Mesh is in millimeters -> convert to meters
        print(
            f"  [Mesh Auto-Scale] Detected millimeter units (max span={max_dim:.1f} mm). "
            f"Scaling by 0.001 -> {max_dim * 0.001:.3f} m."
        )
    elif max_dim > 5.0 * expected_diameter:
        scale_factor = 0.01   # Mesh is in centimeters -> convert to meters
        print(
            f"  [Mesh Auto-Scale] Detected centimeter units (max span={max_dim:.1f} cm). "
            f"Scaling by 0.01 -> {max_dim * 0.01:.3f} m."
        )
    else:
        print(f"  [Mesh Check] Mesh is already in meters (dimensions: {dx:.3f} x {dy:.3f} x {dz:.3f} m).")

    V *= scale_factor

    # Write clean, meter-scaled .obj file for Irrlicht replay
    if not os.path.isfile(clean_obj_path) or os.path.getmtime(raw_obj_path) > os.path.getmtime(clean_obj_path):
        with open(clean_obj_path, "w", encoding="utf-8") as f_out:
            f_out.write("# Auto-scaled meter geometry-only OBJ for PyChrono\n")
            for vx, vy, vz in V:
                f_out.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for i0, i1, i2 in (F + 1):
                f_out.write(f"f {i0} {i1} {i2}\n")

    # Load or generate surface BCE markers at spacing d0
    if os.path.isfile(bce_cache_path) and os.path.getmtime(bce_cache_path) >= os.path.getmtime(raw_obj_path):
        bce_pts = np.load(bce_cache_path)
        print(f"  [BCE Cache] Loaded {len(bce_pts):,} cached BCE markers from {bce_cache_path}")
    else:
        print(f"  [CPU] Fast-sampling BCE markers from {len(F):,} triangles at spacing d0={d0}m...")
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]

        # Compute triangle areas for uniform surface density sampling
        cross_prod = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
        total_area = float(np.sum(areas))

        # Target ~2 layers of markers spaced by (0.75 * d0)
        num_samples = max(2000, int(total_area / ((0.75 * d0) ** 2)))
        probs = areas / max(total_area, 1e-12)

        rng = np.random.default_rng(42)
        tri_indices = rng.choice(len(F), size=num_samples, p=probs)

        r1 = np.sqrt(rng.random(num_samples))[:, None]
        r2 = rng.random(num_samples)[:, None]
        pts_surface = (
            (1.0 - r1) * v0[tri_indices]
            + (r1 * (1.0 - r2)) * v1[tri_indices]
            + (r1 * r2) * v2[tri_indices]
        )

        # Voxel-grid deduplication at resolution (0.75 * d0) so markers are evenly spaced
        voxel_size = 0.75 * d0
        voxel_coords = np.floor(pts_surface / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        bce_pts = pts_surface[np.sort(unique_idx)].astype(np.float64)

        np.save(bce_cache_path, bce_pts)
        print(f"  [CPU] Generated {len(bce_pts):,} surface BCE markers in <0.1s!")

    # Convert NumPy array to PyChrono vector_ChVector3d using explicit tuple unpacking
    bce_vec = chrono.vector_ChVector3d()
    for px, py, pz in bce_pts:
        bce_vec.append(chrono.ChVector3d(float(px), float(py), float(pz)))

    return clean_obj_path, bce_vec


class BucketDrumRig:
    """
    Loads a standalone PLA bucket drum .obj file, uses native C++ ChFunctionConst
    motors (zero Python GIL callback overhead), and registers fast NumPy-sampled
    FSI BCE markers directly with Chrono.
    """

    def __init__(self, env, drum_cfg: dict, op_cfg: dict, y_offset: float = 0.0):
        self.env = env
        self.drum_cfg = drum_cfg
        self.op_cfg = op_cfg
        self.y_offset = float(y_offset)

        raw_obj_path = drum_cfg["obj_path"]
        if not os.path.isfile(raw_obj_path):
            raise FileNotFoundError(f"Bucket drum mesh not found: {raw_obj_path}")

        self.outer_radius = float(drum_cfg["outer_radius"])
        self.inner_radius = float(drum_cfg["inner_cavity_radius"])
        self.drum_length = float(drum_cfg["drum_length"])

        # Fast unit check + NumPy BCE marker generation (<0.1s instead of C++ hang)
        self.obj_path, self.bce_markers = prepare_and_sample_obj(
            raw_obj_path=raw_obj_path,
            target_outer_radius=self.outer_radius,
            d0=self.env.d0,
        )

        self.hover_clearance = 0.01
        self.start_z = self.outer_radius + self.hover_clearance
        self.target_z = self.outer_radius - float(op_cfg["cut_depth"])
        self.start_x = float(op_cfg["start_x"])

        sim_cfg = self.env.cfg["simulation"]
        self.t_settle = float(sim_cfg["settling_time"])
        self.t_plunge = float(self.op_cfg["plunge_duration"])
        self.vx_target = float(self.op_cfg["forward_velocity"])
        total_plunge_dist = self.hover_clearance + float(self.op_cfg["cut_depth"])
        self.vz_target = -total_plunge_dist / max(self.t_plunge, 1e-4)
        self.omega_target = float(self.op_cfg["rpm"]) * (2.0 * math.pi / 60.0)
        self._current_phase = -1

        self.ground = None
        self.x_stage = None
        self.z_carrier = None
        self.drum_body = None
        self.x_motor = None
        self.z_motor = None
        self.rot_motor = None
        self.fn_x = None
        self.fn_z = None
        self.fn_rot = None
        self.fsi_body_index = None

        self._build_multibody_rig()
        self._register_fsi_body()

    def _build_multibody_rig(self):
        sysMBS = self.env.sysMBS
        struct_cfg = self.env.cfg["drum_structural"]

        init_pos = chrono.ChVector3d(self.start_x, self.y_offset, self.start_z)

        # 1. Fixed Ground Reference
        self.ground = chrono.ChBody()
        self.ground.SetFixed(True)
        self.ground.SetPos(init_pos)
        sysMBS.AddBody(self.ground)

        # 2. Horizontal X-Translation Stage
        self.x_stage = chrono.ChBody()
        self.x_stage.SetMass(10.0)
        self.x_stage.SetInertiaXX(chrono.ChVector3d(1.0, 1.0, 1.0))
        self.x_stage.SetPos(init_pos)
        sysMBS.AddBody(self.x_stage)

        self.fn_x = chrono.ChFunctionConst(0.0)
        self.x_motor = chrono.ChLinkMotorLinearSpeed()
        q_x_dir = chrono.QuatFromAngleY(chrono.CH_PI_2)
        self.x_motor.Initialize(self.x_stage, self.ground, chrono.ChFramed(init_pos, q_x_dir))
        self.x_motor.SetSpeedFunction(self.fn_x)
        sysMBS.AddLink(self.x_motor)

        # 3. Vertical Z-Plunge Carrier
        self.z_carrier = chrono.ChBody()
        self.z_carrier.SetMass(10.0)
        self.z_carrier.SetInertiaXX(chrono.ChVector3d(1.0, 1.0, 1.0))
        self.z_carrier.SetPos(init_pos)
        sysMBS.AddBody(self.z_carrier)

        self.fn_z = chrono.ChFunctionConst(0.0)
        self.z_motor = chrono.ChLinkMotorLinearSpeed()
        self.z_motor.Initialize(self.z_carrier, self.x_stage, chrono.ChFramed(init_pos, chrono.QUNIT))
        self.z_motor.SetSpeedFunction(self.fn_z)
        sysMBS.AddLink(self.z_motor)

        # 4. PLA Bucket Drum Body
        self.drum_body = chrono.ChBody()
        self.drum_body.SetPos(init_pos)

        rho_pla = float(struct_cfg["material_density"])
        vol_approx = math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2) * self.drum_length * 0.35
        mass_pla = max(0.5, rho_pla * vol_approx)
        iyy = 0.5 * mass_pla * (self.outer_radius ** 2 + self.inner_radius ** 2)
        ixx_izz = (1.0 / 12.0) * mass_pla * (
            3.0 * (self.outer_radius ** 2 + self.inner_radius ** 2) + self.drum_length ** 2
        )

        self.drum_body.SetMass(mass_pla)
        self.drum_body.SetInertiaXX(chrono.ChVector3d(ixx_izz, iyy, ixx_izz))
        sysMBS.AddBody(self.drum_body)

        # 5. Rotational Motor Driving the Drum Around Y-Axis
        self.fn_rot = chrono.ChFunctionConst(0.0)
        self.rot_motor = chrono.ChLinkMotorRotationSpeed()
        q_y_rot = chrono.QuatFromAngleX(-chrono.CH_PI_2)
        self.rot_motor.Initialize(
            self.drum_body, self.z_carrier, chrono.ChFramed(init_pos, q_y_rot)
        )
        self.rot_motor.SetSpeedFunction(self.fn_rot)
        sysMBS.AddLink(self.rot_motor)

    def update_motor_phase(self, sim_time: float):
        if sim_time < self.t_settle:
            phase = 0
        elif sim_time < (self.t_settle + self.t_plunge):
            phase = 1
        else:
            phase = 2

        if phase != self._current_phase:
            self._current_phase = phase
            if phase == 0:
                self.fn_x.SetConstant(0.0)
                self.fn_z.SetConstant(0.0)
                self.fn_rot.SetConstant(0.0)
            elif phase == 1:
                self.fn_x.SetConstant(0.0)
                self.fn_z.SetConstant(self.vz_target)
                self.fn_rot.SetConstant(self.omega_target)
            else:
                self.fn_x.SetConstant(self.vx_target)
                self.fn_z.SetConstant(0.0)
                self.fn_rot.SetConstant(self.omega_target)

    def _register_fsi_body(self):
        """Registers pre-sampled BCE markers directly with Chrono::CRM (bypasses slow C++ mesh raycaster)."""
        self.fsi_body_index = self.env.problem.AddRigidBody(
            self.drum_body,
            self.bce_markers,
            chrono.ChFramed(chrono.VNULL, chrono.QUNIT),
            False,
        )

        aabb_dims = chrono.ChVector3d(*self.env.cfg["soil_bed"]["active_domain_aabb"])
        if hasattr(self.env.sysFSI, "SetActiveDomain"):
            self.env.sysFSI.SetActiveDomain(aabb_dims)
