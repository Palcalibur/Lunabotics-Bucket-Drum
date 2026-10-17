import chrono_setup
import pychrono.fsi as fsi

# Look for CRM material-property types and rheology enums.
for name in dir(fsi):
    if any(term in name.lower() for term in ("soilproperties", "rheologycrm")):
        obj = getattr(fsi, name)
        print(f"\n{name}: {obj}")
        print("Relevant attributes:")
        print([
            attr for attr in dir(obj)
            if any(term in attr.lower() for term in
                   ("rheology", "density", "friction", "diam", "young", "poisson"))
        ])

# Check whether the CRM setter is exposed on the actual fluid-system object.
import pychrono as chrono
import pychrono.vehicle as vehicle

sys = chrono.ChSystemNSC()
terrain = vehicle.CRMTerrain(sys, 0.05)
fluid = terrain.GetFluidSystemSPH()

print("\nSetCrmSPH on terrain:", hasattr(terrain, "SetCrmSPH"))
print("SetCrmSPH on fluid:", hasattr(fluid, "SetCrmSPH"))
print("Fluid-system type:", type(fluid))