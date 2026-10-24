import os
import yaml
import pychrono as chrono

# Support both pychrono.fsi and pychrono.fsi.sph namespace layouts in newest builds
try:
    import pychrono.fsi.sph as fsi_sph
except ImportError:
    import pychrono.fsi as fsi_sph


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CRMExcavationEnvironment:
    """
    Manages the coupled ChSystemSMC multibody system and GPU-accelerated
    ChFsiProblemCartesian (Chrono::CRM) BP-1 lunar regolith simulant pit.
    """

    def __init__(self, crm_config: dict, verbose: bool = False):
        self.cfg = crm_config
        self.verbose = verbose

        # 1. Initialize Multibody System (SMC) with Multi-Core Threading
        self.sysMBS = chrono.ChSystemSMC()
        self.sysMBS.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        num_threads = min(8, os.cpu_count() or 4)
        self.sysMBS.SetNumThreads(num_threads, num_threads, num_threads)

        g_vec = chrono.ChVector3d(*self.cfg["environment"]["gravity"])
        self.sysMBS.SetGravitationalAcceleration(g_vec)

        # 2. Initialize FSI-SPH Cartesian Problem (CUDA GPU Solver)
        self.d0 = float(self.cfg["sph_discretization"]["initial_spacing"])
        self.problem = fsi_sph.ChFsiProblemCartesian(self.d0, self.sysMBS)
        self.problem.SetVerbose(self.verbose)

        self.sysFSI = self.problem.GetFsiSystemSPH()
        self.sysSPH = self.problem.GetFluidSystemSPH()

        self._configure_crm_parameters()

    def _configure_crm_parameters(self):
        soil = self.cfg["soil_material"]
        sph = self.cfg["sph_discretization"]
        sim = self.cfg["simulation"]

        g_vec = chrono.ChVector3d(*self.cfg["environment"]["gravity"])
        self.problem.SetGravitationalAcceleration(g_vec)
        self.problem.SetStepSizeCFD(float(sim["dt_cfd"]))
        self.problem.SetStepsizeMBD(float(sim["dt_mbd"]))

        # Configure BP-1 Soil Properties for CRM
        soil_props = fsi_sph.SoilProperties()
        soil_props.density = float(soil["bulk_density"])
        soil_props.Young_modulus = float(soil["young_modulus"])
        soil_props.Poisson_ratio = float(soil["poisson_ratio"])
        soil_props.cohesion_coeff = float(soil["cohesion"])
        soil_props.mu_I0 = float(soil["I_0"])
        soil_props.mu_fric_s = float(soil["mu_s"])
        soil_props.mu_fric_2 = float(soil["mu_2"])
        soil_props.average_diam = float(soil["average_grain_diameter"])

        if hasattr(fsi_sph, "RheologyCRM_MU_OF_I"):
            soil_props.rheology_model = fsi_sph.RheologyCRM_MU_OF_I

        self.problem.SetCrmSPH(soil_props)

        # Configure SPH Discretization Parameters directly on sysSPH
        self.sysSPH.SetInitialSpacing(float(self.d0))
        self.sysSPH.SetKernelMultiplier(float(sph["kernel_multiplier"]))
        self.sysSPH.SetNumBCELayers(int(sph["num_bce_layers"]))
        self.sysSPH.SetArtificialViscosityCoefficient(float(sph["artificial_viscosity"]))

        if hasattr(fsi_sph, "ShiftingMethod_PPST_XSPH"):
            self.sysSPH.SetShiftingMethod(fsi_sph.ShiftingMethod_PPST_XSPH)
        else:
            self.sysSPH.SetShiftingMethod(fsi_sph.ShiftingMethod_XSPH)

        self.sysSPH.SetShiftingXSPHParameters(float(sph["xsph_coefficient"]))

    def build_regolith_pit(self):
        """
        Constructs the BP-1 SPH particle bed on host and copies arrays to CUDA GPU.
        MUST be called AFTER rigid bodies are registered with self.problem.AddRigidBody().
        """
        bed_cfg = self.cfg["soil_bed"]
        box_size = chrono.ChVector3d(*bed_cfg["dimensions"])
        box_center = chrono.ChVector3d(*bed_cfg["center"])

        est_particles = int((box_size.x * box_size.y * box_size.z) / (self.d0 ** 3))
        print(f"  [CPU] Generating ~{est_particles:,} SPH particles and BCE markers in host RAM...")

        side_flags = fsi_sph.BoxSide_ALL & ~fsi_sph.BoxSide_Z_POS
        self.problem.Construct(box_size, box_center, side_flags)

        print("  [GPU] Transferring SPH buffers to VRAM and initializing CUDA kernels...")
        self.problem.Initialize()
        print(
            f"  [GPU] CUDA Ready! Active SPH particles: {self.sysSPH.GetNumFluidMarkers():,} | "
            f"BCE markers: {self.sysSPH.GetNumBoundaryMarkers():,}"
        )

    def step(self, dt: float):
        """Advances the coupled MBS + CUDA SPH system by dt."""
        self.problem.DoStepDynamics(dt)
