"""Reaction-eikonal on a biventricular mesh, matching
benchmark/opencarp/reaction_eikonal/parameters.par.

    python demo/eikonal_ventricle.py [--re-plus] [--ms N]

--re-plus is RE+ (openCARP dream.solve = 1, its +F re_plus.par); the default is
RE-.  --ms N shortens the run, as openCARP's -tend N does.  Compare the two with
benchmark/opencarp/reaction_eikonal/compare.py.
"""

import sys
from pathlib import Path

import torch

import torchcor as tc
from torchcor.ionic import TenTusscherPanfilov
from torchcor.simulator import ReactionEikonal

tc.set_device("cuda:0")
dtype = tc.float64                              # openCARP is double precision
dt = 0.01                                       # ms
snapshot_interval = 1                           # ms
simulation_time = 500.0                         # ms
if "--ms" in sys.argv:
    simulation_time = float(sys.argv[sys.argv.index("--ms") + 1])
diffusion = "--re-plus" in sys.argv

mesh_dir = Path.home()/"Data/ventricle/Case_1"
result_path = (Path(__file__).resolve().parent
               / ("biventricle_eikonal_rep" if diffusion else "biventricle_eikonal"))

im = TenTusscherPanfilov(dt, cell_type="ENDO", dtype=dtype)   # defaults to EPI
simulator = ReactionEikonal(ionic_models=[im], T=simulation_time, dt=dt,
                            diffusion=diffusion, dtype=dtype)
simulator.load_mesh(path=mesh_dir, unit_conversion=1000)      # um -> mm

# Conduction velocities in mm/ms; the endocardial layer (44, 45, 46) is faster
# than the bulk myocardium (34, 35).
simulator.add_velocity([34, 35],     vel_l=0.60, vel_t=0.38)
simulator.add_velocity([44, 45, 46], vel_l=0.90, vel_t=0.41)
if diffusion:
    # Not calibrated to reproduce the velocities above; diffusion loads the
    # prescribed front rather than setting it.
    simulator.add_conductivity(region_ids=None, il=0.174, it=0.019,
                               el=0.625, et=0.236)

# His-Purkinje junctions: LV fascicles at 0 ms, RV at 5 ms.
for name, start in (("LV_sf", 0.0), ("LV_pf", 0.0), ("LV_af", 0.0),
                    ("RV_sf", 5.0), ("RV_mod", 5.0)):
    simulator.add_stimulus(mesh_dir/f"{name}.vtx", start=start,
                           duration=1.0, intensity=100)

AT = simulator.eikonal_activation_times()       # prescribed arrival times, not LAT
print(f"eikonal AT: {AT.min().item():.3f} .. {AT.max().item():.3f} ms", flush=True)

Vm = simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100,   # tolerances: RE+ only
                     snapshot_interval=snapshot_interval,
                     verbose=True, result_path=result_path)

result_path.mkdir(parents=True, exist_ok=True)
torch.save(AT.cpu(), result_path/"act.pt")
torch.save(Vm.cpu(), result_path/"vm.pt")
print(f"saved {tuple(Vm.shape)} to {result_path}", flush=True)
