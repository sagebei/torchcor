import torch
import torchcor as tc
from torchcor.electrophysiology import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from torchcor.electrocardiogram import LeadField
from pathlib import Path

# Specify the GPU device for running the simulation
tc.set_device("cuda:0")
device = torch.device("cuda:0")
dtype = tc.float64
# The total simulation duration (ms)
simulation_time = 500
dt = 0.01

# Base data directory of the coupled heart-torso model. Expected layout:
#   <base>/heart       : standalone heart mesh (.pts .elem .lon) + pacing/*.vtx
#   <base>/mesh        : full torso mesh (heart embedded as tagged regions)
#   <base>/electrodes  : lf_src.vtx  (V1-V6, RA, LA, RL, LL)
base_dir       = Path.home() / "Data/torso/HC2"
heart_mesh_dir = base_dir / "heart"
torso_mesh_dir = base_dir / "mesh"
electrode_file = base_dir / "electrodes" / "lf_src.vtx"

# =====================================================================
# STEP 1: EP simulation on the heart mesh -> transmembrane potential Vm
#   mass_lumping=True matches openCARP's mass-lumped discretisation,
#   which gives the correct conduction velocity (and ECG morphology).
# =====================================================================
im = TenTusscherPanfilov(cell_type="EPI", dt=dt, dtype=dtype)
simulator = Monodomain(ionic_models=[im], T=simulation_time, dt=dt,
                       dtype=dtype, mass_lumping=True)
simulator.load_mesh(path=heart_mesh_dir)
simulator.add_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
simulator.add_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)
for name, start in [("LV_sf", 0.0), ("LV_pf", 0.0), ("LV_af", 0.0),
                    ("RV_sf", 5.0), ("RV_mod", 5.0)]:
    simulator.add_stimulus(heart_mesh_dir / "pacing" / f"{name}.vtx",
                           start=start, duration=1.0, intensity=100)
Vm = simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100,
                     snapshot_interval=1, verbose=True,
                     result_path="./ecg_demo").cpu()   # Vm: (T, N_heart)

# Free the EP solver's GPU memory before assembling the lead-field matrices.
del simulator, im
torch.cuda.empty_cache()

# =====================================================================
# STEP 2: Build the lead-field solver on the coupled heart-torso mesh
# =====================================================================
lf = LeadField(torso_mesh_dir, heart_mesh_dir, device=device, dtype=dtype)
# Passive torso / organ conductivities (S/m, isotropic)
lf.add_torso_conductivity([20, 21], g=0.25)
lf.add_torso_conductivity([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23], g=0.6667)
lf.add_torso_conductivity([3], g=0.05)
lf.add_torso_conductivity([1, 7], g=0.2472)
lf.add_torso_conductivity([4], g=0.1667)
lf.add_torso_conductivity([5], g=0.1667)
lf.add_torso_conductivity([6], g=0.0714)
lf.add_torso_conductivity([2], g=0.117)
lf.add_torso_conductivity([8], g=0.1)
lf.add_torso_conductivity([9], g=0.1)
# Anisotropic bidomain heart conductivities (intra/extra, S/m)
lf.add_heart_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
lf.add_heart_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)
lf.build()

# =====================================================================
# STEP 3: Precompute one lead field per electrode, then the 12-lead ECG
# =====================================================================
lf.load_electrodes(electrode_file)   # V1-V6, RA, LA, RL, LL  (RL = reference)
lf.precompute_all()

ecg = lf.compute_12lead(Vm)          # dict: I, II, III, aVR, aVL, aVF, V1..V6
lf.plot_ecg(ecg, filename="ecg_12lead.png", dt=1.0)
print("Saved 12-lead ECG to ecg_12lead.png")
