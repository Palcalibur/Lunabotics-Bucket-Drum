import os
import math
import pychrono as chrono
from typing import Any, cast

class PlungeAndTraverseFunction(chrono.ChFunction):
    """Custom velocity function for linear motors during settling, plunge, and excavation."""

    def __init__(self, t_start: float, t_end: float, velocity: float):
        super().__init__()
        self.t_start = t_start
        self.t_end = t_end
        self.velocity = velocity

    def GetVal(self, x: float) -> float:
        if self.t_start <= x <= self.t_end:
            return self.velocity
        return 0.0

    def Clone(self):
        return PlungeAndTraverseFunction(self.t_start, self.t_end, self.velocity)


class BucketDrumRig:
    """
    Loads a standalone PLA bucket drum .obj file, attaches kinematic actuators for
    vertical plunge (Z), forward traverse (X), and drum rotation (about Y),
    and registers its surface BCE markers with the Chrono::CRM FSI system.
    """

    def __init__(self, env, drum_cfg: dict, op_cfg: dict, y_offset: float = 0.0):
        self.env = env
        self.drum_cfg = drum_cfg
        self.op_cfg = op_cfg
        self.y_offset = float(y_offset)

        self.obj_path = drum_cfg["obj_path"]
        if not os.path.isfile(self.obj_path):
            raise FileNotFoundError(f"Bucket drum mesh not found: {self.obj_path}")

        self.outer_radius = float(drum_cfg["outer_radius"])
        self.inner_radius = float(drum_cfg["inner_cavity_radius"])
        self.drum_length = float(drum_cfg["drum_length"])

        # Calculate starting Z height: hovers 1 cm above Z=0 soil bed before plunging
        self.hover_clearance = 0.01
        self.start_z = self.outer_radius + self.hover_clearance
        self.target_z = self.outer_radius - float(op_cfg["cut_depth"])
        self.start_x = float(op_cfg["start_x"])

        self.ground = None
        self.x_stage = None
        self.z_carrier = None
        self.drum_body = None
        self.x_motor = None
        self.z_motor = None
        self.rot_motor = None
        self.fsi_body_index = None

        self._build_multibody_rig()
        self._register_fsi_body()

    def _build_multibody_rig(self):
        sysMBS = self.env.sysMBS
        sim_cfg = self.env.cfg["simulation"]
        struct_cfg = self.env.cfg["drum_structural"]

        t_settle = float(sim_cfg["settling_time"])
        t_plunge = float(self.op_cfg["plunge_duration"])
        t_total = float(sim_cfg["duration"])

        # Place the drum at the configured (start_x, y_offset, start_z) path coordinate
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

        self.x_motor = chrono.ChLinkMotorLinearSpeed()
        # ChLinkMotorLinear guides along the Z-axis of the motor frame; rotate Y by +90 deg for X-axis
        q_x_dir = chrono.QuatFromAngleY(chrono.CH_PI_2)
        self.x_motor.Initialize(self.x_stage, self.ground, chrono.ChFramed(init_pos, q_x_dir))
        vx = float(self.op_cfg["forward_velocity"])
        self.x_motor.SetSpeedFunction(
            PlungeAndTraverseFunction(t_settle + t_plunge, t_total, vx)
        )
        sysMBS.AddLink(self.x_motor)

        # 3. Vertical Z-Plunge Carrier
        self.z_carrier = chrono.ChBody()
        self.z_carrier.SetMass(10.0)
        self.z_carrier.SetInertiaXX(chrono.ChVector3d(1.0, 1.0, 1.0))
        self.z_carrier.SetPos(init_pos)
        sysMBS.AddBody(self.z_carrier)

        self.z_motor = chrono.ChLinkMotorLinearSpeed()
        # Motor Z-axis is already aligned with global Z
        self.z_motor.Initialize(self.z_carrier, self.x_stage, chrono.ChFramed(init_pos, chrono.QUNIT))
        total_plunge_dist = self.hover_clearance + float(self.op_cfg["cut_depth"])
        vz_plunge = -total_plunge_dist / max(t_plunge, 1e-4)
        self.z_motor.SetSpeedFunction(
            PlungeAndTraverseFunction(t_settle, t_settle + t_plunge, vz_plunge)
        )
        sysMBS.AddLink(self.z_motor)

        # 4. PLA Bucket Drum Mesh Body
        contact_mat = chrono.ChContactMaterialSMC()
        contact_mat.SetYoungModulus(float(struct_cfg["contact_young_modulus"]))
        contact_mat.SetFriction(float(struct_cfg["contact_friction"]))

        self.drum_body = chrono.ChBodyEasyMesh(
            self.obj_path,
            float(struct_cfg["material_density"]),
            True,   # compute mass, inertia, COG from mesh
            True,   # visualization asset
            False,  # MBS collision disabled (handled by FSI BCE markers)
            contact_mat,
            float(struct_cfg["mesh_swept_sphere_radius"]),
        )
        self.drum_body.SetPos(init_pos)
        sysMBS.AddBody(self.drum_body)

        # 5. Rotational Motor Driving the Drum Around Y-Axis
        self.rot_motor = chrono.ChLinkMotorRotationSpeed()
        # Rotate motor frame X by -90 deg so motor Z-axis aligns with global Y-axis
        q_y_rot = chrono.QuatFromAngleX(-chrono.CH_PI_2)
        self.rot_motor.Initialize(
            self.drum_body, self.z_carrier, chrono.ChFramed(init_pos, q_y_rot)
        )
        omega_rad_s = float(self.op_cfg["rpm"]) * (2.0 * math.pi / 60.0)
        self.rot_motor.SetSpeedFunction(
            PlungeAndTraverseFunction(t_settle, t_total, omega_rad_s)
        )
        sysMBS.AddLink(self.rot_motor)

    def _register_fsi_body(self):
        """Generates surface BCE markers on the .obj mesh and enables FSI active domain."""
        trimesh_shape = chrono.TrimeshShape(
            chrono.VNULL, chrono.QUNIT, self.obj_path, 1.0, 0.0
        )
        geom = cast(Any, chrono.ChBodyGeometry())
        geom.coll_meshes.push_back(trimesh_shape)

        # check_embedded=True removes any soil particles overlapping the drum at t=0
        self.fsi_body_index = self.env.problem.AddRigidBody(
            self.drum_body, geom, True, False
        )

        # Set active domain AABB around the drum for GPU speed
        aabb_dims = chrono.ChVector3d(*self.env.cfg["soil_bed"]["active_domain_aabb"])
        if hasattr(self.env.sysFSI, "SetActiveDomain"):
            self.env.sysFSI.SetActiveDomain(aabb_dims)