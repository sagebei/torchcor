"""
Full pipeline: cardiac EP simulation (monodomain) -> ECG lead-field forward
problem -> 12-lead ECG, saved to disk.  Run `match.py` afterwards to overlay
the result against the openCARP ground truth.

Key point for matching openCARP: the monodomain is run with mass_lumping=True.
openCARP lumps the mass matrix by default (mass_lumping=1); torchcor's default
consistent mass matrix conducts ~35% too fast, which narrows the QRS and
distorts the ECG.  With lumping, total ventricular activation (~98 ms) and the
12-lead morphology line up with openCARP.
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Resolve imports against THIS repo so the edited torchcor (with mass_lumping)
# is used instead of any pip-installed copy in site-packages.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[2]))   # .../           -> `import torchcor` == this repo
os.chdir(_HERE.parent)

import torchcor as tc
from torchcor.electrophysiology import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from torchcor.electrocardiogram import LeadField

# ---------- Config ----------
device = torch.device("cuda:0")
dtype = torch.float64
tc.set_device("cuda:0")

# Base data directory -- all other paths are derived from it.
base_dir       = Path("/data/Bei/Torso/HC2")
mesh_dir       = base_dir / "heart"
torso_mesh_dir = base_dir / "mesh"
heart_mesh_dir = base_dir / "heart"
electrode_file = base_dir / "electrodes" / "lf_src.vtx"
opencarp_ecg   = base_dir / "opencarp_results" / "SIM_EP" / "ECG_lead_field.dat"
result_dir     = base_dir / "torchcor_results"

vm_file = result_dir / "Vm.pt"

# ========== STEP 1: EP simulation (mass lumping ON to match openCARP) ==========
if vm_file.exists():
    print(f"Loading saved Vm from {vm_file}")
    Vm = torch.load(vm_file, weights_only=True)
else:
    print("=" * 60)
    print("STEP 1: Cardiac EP simulation (500 ms, mass_lumping=True)")
    print("=" * 60)
    im = TenTusscherPanfilov(cell_type="EPI", dt=0.01, dtype=dtype)
    simulator = Monodomain(ionic_models=[im], T=500, dt=0.01, dtype=dtype,
                           mass_lumping=True)
    simulator.load_mesh(path=mesh_dir)
    simulator.add_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
    simulator.add_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)
    for name, start in [("LV_sf", 0.0), ("LV_pf", 0.0), ("LV_af", 0.0),
                        ("RV_sf", 5.0), ("RV_mod", 5.0)]:
        simulator.add_stimulus(mesh_dir / "pacing" / f"{name}.vtx",
                               start=start, duration=1.0, intensity=100)
    Vm = simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100,
                         snapshot_interval=1, verbose=True,
                         result_path=result_dir).cpu()
    torch.save(Vm, vm_file)
    print(f"Saved Vm to {vm_file}")
    del simulator, im
    torch.cuda.empty_cache()
print(f"Vm shape: {tuple(Vm.shape)}")

# ========== STEP 2: build lead-field solver ==========
print("\n" + "=" * 60)
print("STEP 2: Assembling ECG lead-field matrices")
print("=" * 60)
lf = LeadField(torso_mesh_dir, heart_mesh_dir, device=device, dtype=dtype)
# Torso conductivities (S/m, isotropic) -- from openCARP monodomain.par g_bath
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
# Heart conductivities (S/m, anisotropic bidomain)
lf.add_heart_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
lf.add_heart_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)
lf.build()

# ========== STEP 3: load electrodes & precompute lead fields ==========
print("\n" + "=" * 60)
print("STEP 3: Precomputing lead fields (9 CG solves)")
print("=" * 60)
# lf_src.vtx order: V1, V2, V3, V4, V5, V6, RA, LA, RL, LL  (RL = reference)
lf.load_electrodes(electrode_file)
lf.precompute_all(a_tol=1e-8, r_tol=1e-8, max_iter=20000)

# ========== STEP 4: compute & save 12-lead ECG ==========
print("\n" + "=" * 60)
print("STEP 4: Computing & saving 12-lead ECG")
print("=" * 60)
ecg = lf.compute_12lead(Vm.cpu())

result_dir.mkdir(parents=True, exist_ok=True)
ecg_file = result_dir / "torchcor_ecg.npz"
np.savez(ecg_file, dt_ms=1.0, **{lead: sig.cpu().numpy() for lead, sig in ecg.items()})
print(f"Saved torchcor 12-lead ECG to {ecg_file}")
print("Run `python match.py` to overlay it against the openCARP ground truth.")
print("Done.")
