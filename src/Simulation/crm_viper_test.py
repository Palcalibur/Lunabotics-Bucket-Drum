import chrono_setup
import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import cupy as cp  # CUDA offloading library (pip install cupy-cuda12x)

# -----------------------------------------------------------------------------
# 1. Simulation & Particle Parameters
# -----------------------------------------------------------------------------
NUM_PARTICLES = 300
PARTICLE_RADIUS = 0.05
BOX_HALF_SIZE = 1.5
GRAVITY = cp.array([0.0, 0.0, -9.81], dtype=cp.float32)
RESTITUTION = 0.6
DAMPING = 0.99
TIMESTEP = 0.01

# Initialize particle state on GPU memory (CUDA)
# Random positions inside a bounding area above the ground
pos_gpu = cp.random.uniform(
    low=[-0.8, -0.8, 1.0],
    high=[0.8, 0.8, 2.5],
    size=(NUM_PARTICLES, 3)
).astype(cp.float32)

vel_gpu = cp.zeros((NUM_PARTICLES, 3), dtype=cp.float32)

# -----------------------------------------------------------------------------
# 2. Setup Chrono System & Irrlicht GUI
# -----------------------------------------------------------------------------
sys = chrono.ChSystemNSC()

# Create fixed container ground / walls for visualization
box_body = chrono.ChBody()
box_body.SetFixed(True)
box_shape = chrono.ChVisualShapeBox(BOX_HALF_SIZE * 2, BOX_HALF_SIZE * 2, 0.1)
box_shape.SetColor(chrono.ChColor(0.2, 0.2, 0.2))
box_body.AddVisualShape(box_shape, chrono.ChFramed(chrono.ChVector3d(0, 0, -0.05)))
sys.Add(box_body)

# Create visual spheres in Chrono to mirror GPU particle states
chrono_particles = []
pos_host = cp.asnumpy(pos_gpu)

for i in range(NUM_PARTICLES):
    body = chrono.ChBody()
    body.SetFixed(True)  # Positions updated manually from GPU
    body.SetPos(chrono.ChVector3d(float(pos_host[i, 0]), float(pos_host[i, 1]), float(pos_host[i, 2])))

    sphere_shape = chrono.ChVisualShapeSphere(PARTICLE_RADIUS)
    sphere_shape.SetColor(chrono.ChColor(0.2, 0.6, 0.9))
    body.AddVisualShape(sphere_shape)

    sys.Add(body)
    chrono_particles.append(body)

# Initialize Irrlicht Visual System
vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(sys)
vis.SetWindowSize(1280, 720)
vis.SetWindowTitle("CUDA Particle Simulation with PyChrono Irrlicht")
vis.Initialize()
vis.AddLogo(chrono.GetChronoDataFile("logo_chrono_alpha.png"))
vis.AddSkyBox()
vis.AddCamera(chrono.ChVector3d(0, -3.5, 2.0), chrono.ChVector3d(0, 0, 1.0))
vis.AddLightDirectional()

# -----------------------------------------------------------------------------
# 3. CUDA GPU Update Kernel (Vectorized on Device)
# -----------------------------------------------------------------------------
def advance_particles_cuda(pos, vel, dt):
    # Integrate gravity
    vel += GRAVITY * dt
    pos += vel * dt

    # Boundary collision (Z ground floor, index 2)
    floor_mask = pos[:, 2] < PARTICLE_RADIUS
    pos[floor_mask, 2] = PARTICLE_RADIUS
    vel[floor_mask, 2] = -vel[floor_mask, 2] * RESTITUTION
    vel[floor_mask] *= DAMPING

    # Side boundaries for X (axis 0) and Y (axis 1)
    for axis in [0, 1]:
        pos_wall = pos[:, axis] > (BOX_HALF_SIZE - PARTICLE_RADIUS)
        neg_wall = pos[:, axis] < (-BOX_HALF_SIZE + PARTICLE_RADIUS)

        pos[pos_wall, axis] = BOX_HALF_SIZE - PARTICLE_RADIUS
        vel[pos_wall, axis] = -vel[pos_wall, axis] * RESTITUTION

        pos[neg_wall, axis] = -BOX_HALF_SIZE + PARTICLE_RADIUS
        vel[neg_wall, axis] = -vel[neg_wall, axis] * RESTITUTION

    return pos, vel

# -----------------------------------------------------------------------------
# 4. Main Simulation & Rendering Loop
# -----------------------------------------------------------------------------
while vis.Run():
    # 1. Advance physics entirely on CUDA GPU
    pos_gpu, vel_gpu = advance_particles_cuda(pos_gpu, vel_gpu, TIMESTEP)

    # 2. Copy back positions to update Irrlicht visual proxies
    pos_host = cp.asnumpy(pos_gpu)
    for i in range(NUM_PARTICLES):
        chrono_particles[i].SetPos(
            chrono.ChVector3d(float(pos_host[i, 0]), float(pos_host[i, 1]), float(pos_host[i, 2]))
        )

    # 3. Render frame in Irrlicht
    vis.BeginScene()
    vis.Render()
    vis.EndScene()