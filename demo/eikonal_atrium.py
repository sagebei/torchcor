import torchcor as tc
from torchcor.electrophysiology import ReactionEikonal
from torchcor.ionic import ModifiedMS2v
from pathlib import Path

# Specify the GPU device to run the simulation on
tc.set_device("cuda:0")
dtype = tc.float32
simulation_time = 500      # total duration (ms)
dt = 0.01                  # time step (ms)

# Mitchell-Schaeffer-type atrial cell model
im = ModifiedMS2v(dt, dtype=dtype)
im.u_gate = 0.1
im.u_crit = 0.1
im.tau_in = 0.15
im.tau_out = 1.5
im.tau_open = 105.0
im.tau_close = 185.0

case_name = "Case_18"
mesh_dir = Path("/home/bzhou6/Data/atrium/") / case_name

# 1. Reaction-Eikonal model.  diffusion=False (R-E-): no diffusion/linear solve --
#    each cell fires its own AP as the wavefront arrives, so Vm is cheap to make.
#    (diffusion=True adds electrotonic coupling but costs ~monodomain on a fine mesh.)
simulator = ReactionEikonal(ionic_models=[im], T=simulation_time, dt=dt,
                            diffusion=False, dtype=dtype)
# 2. Load the mesh (.pts .elem .lon)
simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
# 3. Conduction velocities (m/s) per region -- the eikonal times the wavefront.
simulator.add_velocity(region_ids=[1, 2, 3, 4, 5, 6], vel_l=0.6, vel_t=0.3)
# 4. Seed the wavefront (.vtx sites + onset time).
simulator.add_stimulus(f"{mesh_dir}/{case_name}.vtx", start=0.0, duration=2.0, intensity=50)

# 5. (Optional) Eikonal activation times t_a(x) -- the fast standalone activation
#    map.  solve() below reuses this.
AT_eikonal = simulator.eikonal_activation_times()
print("eikonal AT: ", AT_eikonal.min().item(), AT_eikonal.cpu().max().item(), flush=True)

# 6. Reaction -> full Vm.  No linear solve (diffusion=False); a_tol/r_tol/max_iter
#    are R-E+ only.
snapshot_interval = 1
Vm = simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100,
                     snapshot_interval=snapshot_interval, verbose=False,
                     result_path="./eikonal_atrium")

# Vm is the cheap cardiac source: feed it to a lead-field / extracellular solver
# for electrograms and the 12-lead ECG (Neic 2017).  That solve is a separate step.
simulator.save_vm(Vm)              # -> ./eikonal_atrium/Vm.pt
print("saved Vm for ECG:", tuple(Vm.shape), "| range",
      round(Vm.min().item(), 1), round(Vm.max().item(), 1), flush=True)
