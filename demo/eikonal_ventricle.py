import torchcor as tc
from torchcor.simulator import ReactionEikonal
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

# Specify the GPU device to run the simulation on
tc.set_device("cuda:0")
dtype = tc.float32
simulation_time = 500      # total duration (ms)
dt = 0.01                  # time step (ms)

mesh_dir = Path.home() / "Data/ventricle/Case_1"
# TenTusscherPanfilov ventricular cell model
im = TenTusscherPanfilov(cell_type="ENDO", dt=dt, dtype=dtype)

# 1. Reaction-Eikonal model.  diffusion=False (R-E-): no diffusion/linear solve --
#    each cell fires its own AP as the wavefront arrives, so Vm is cheap to make.
#    (diffusion=True adds electrotonic coupling but costs ~monodomain on a fine mesh.)
simulator = ReactionEikonal(ionic_models=[im], 
                            T=simulation_time, 
                            dt=dt,
                            diffusion=True, 
                            dtype=dtype)
# 2. Load the mesh (.pts .elem .lon)
simulator.load_mesh(path=mesh_dir)
# 3. Conduction velocities (m/s) per region -- the eikonal times the wavefront.
#    Fast endocardial layer (44,45,46) a little quicker than bulk myocardium (34,35).
simulator.add_velocity([34, 35],     vel_l=0.60, vel_t=0.38)
simulator.add_velocity([44, 45, 46], vel_l=0.68, vel_t=0.41)
# 4. Seed the wavefront at the His-Purkinje junctions: LV fascicles at t=0, RV at 5 ms.
simulator.add_stimulus(mesh_dir / "LV_sf.vtx",  start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "LV_pf.vtx",  start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "LV_af.vtx",  start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "RV_sf.vtx",  start=5.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

# 5. (Optional) Eikonal activation times t_a(x) -- the fast standalone activation
#    map.  solve() below reuses this.
AT_eikonal = simulator.eikonal_activation_times()
print("eikonal AT: ", AT_eikonal.min().item(), AT_eikonal.cpu().max().item(), flush=True)

# 6. Reaction -> full Vm.  No linear solve (diffusion=False); a_tol/r_tol/max_iter
#    are R-E+ only.
snapshot_interval = 1
Vm = simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100,
                     snapshot_interval=snapshot_interval, 
                     verbose=False,
                     result_path="./biventricle_eikonal")


# simulator.save_vm(Vm)              # -> ./biventricle_eikonal/Vm.pt
# print("saved Vm for ECG:", tuple(Vm.shape), "| range",
#       round(Vm.min().item(), 1), round(Vm.max().item(), 1), flush=True)
