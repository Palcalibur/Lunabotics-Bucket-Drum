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
    Manages the coupled ChSystemSMC multibody system and ChFsiProblemCartesian
    (Chrono::CRM) BP-1 lunar regolith simulant pit.
    """

    def __init__(self, crm_config: dict, verbose: bool = False):
        self.cfg = crm_config
        self.verbose = verbose

        # 1. Initialize Multibody System (SMC)
        self.sysMBS = chrono.ChSystemSMC()
        self.sysMBS.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        g_vec = chrono.ChVector3d(*self.cfg["environment"]["gravity"])
        self.sysMBS.SetGravitationalAcceleration(g_vec)

        # 2. Initialize FSI-SPH Cartesian Problem (CRM)
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
        # NOTE: Errors given by the IDE do NOT show up during runtime
        # the "errors" are linter problems
        soil_props = fsi_sph.SoilProperties()
        soil_props.density = float(soil["bulk_density"])
        soil_props.Young_modulus = float(soil["young_modulus"])
        soil_props.Poisson_ratio = float(soil["poisson_ratio"])
        soil_props.cohesion_coeff = float(soil["cohesion"])  # Renamed from .cohesion
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

        # Prefer PPST_XSPH if available in this Chrono build, otherwise fallback to XSPH
        if hasattr(fsi_sph, "ShiftingMethod_PPST_XSPH"):
            self.sysSPH.SetShiftingMethod(fsi_sph.ShiftingMethod_PPST_XSPH)
        else:
            self.sysSPH.SetShiftingMethod(fsi_sph.ShiftingMethod_XSPH)

        self.sysSPH.SetShiftingXSPHParameters(float(sph["xsph_coefficient"]))

    def build_regolith_pit(self):
        """
        Constructs the BP-1 SPH particle bed and rigid container walls (all sides except top Z+).
        MUST be called AFTER rigid bodies are registered with self.problem.AddRigidBody().
        """
        bed_cfg = self.cfg["soil_bed"]
        box_size = chrono.ChVector3d(*bed_cfg["dimensions"])
        box_center = chrono.ChVector3d(*bed_cfg["center"])

        # Enable BCE walls on bottom and all 4 vertical sides; leave top open
        side_flags = fsi_sph.BoxSide_ALL & ~fsi_sph.BoxSide_Z_POS

        self.problem.Construct(box_size, box_center, side_flags)
        self.problem.Initialize()

    def step(self, dt: float):
        """Advances the coupled MBS + SPH system by dt."""
        self.problem.DoStepDynamics(dt)