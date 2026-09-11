import torchcor as tc
from torchcor.electrophysiology import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path
import torch

# ---------- Config ----------
dtype = tc.float64
tc.set_device("cuda:0")

# Base data directory -- all other paths are derived from it.
base_dir       = Path("/data/Bei/Torso/HC2")
mesh_dir       = base_dir / "heart"
torso_mesh_dir = base_dir / "mesh"
heart_mesh_dir = base_dir / "heart"
electrode_file = base_dir / "electrodes" / "lf_src.vtx"
result_dir     = base_dir / "torchcor_results"

vm_file = result_dir / "Vm.pt"

simulation_time = 500
dt = 0.01
snapshot_interval = 1

# openCARP monodomain.par uses TT2 with "flags=EPI" for all 5 tags
# (no transmural heterogeneity).  Match that here.
im = TenTusscherPanfilov(cell_type="EPI", dt=dt, dtype=dtype)
simulator = Monodomain(ionic_models=[im], T=simulation_time, dt=dt, dtype=dtype, mass_lumping=True)
simulator.load_mesh(path=mesh_dir)
simulator.add_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
simulator.add_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

simulator.add_stimulus(mesh_dir / "pacing" / "LV_sf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "LV_pf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "LV_af.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "RV_sf.vtx", start=5.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

Vm = simulator.solve(
    a_tol=1e-5, r_tol=1e-5, max_iter=100,
    snapshot_interval=snapshot_interval,
    verbose=True,
    result_path=result_dir,
)
print(f"Vm shape: {Vm.shape}")
Vm = Vm.cpu()
torch.save(Vm, vm_file)
