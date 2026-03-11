import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v, CourtemancheRamirezNattel
from pathlib import Path

# Specify the GPU device to run the simulation on 
tc.set_device("cuda:0")
dtype = tc.float32
# The total simulation duration (ms)
simulation_time = 500
# time interval
dt = 0.01

# Load in the ionic model, such as MitchellSceaffer, CourtemancheRamirezNattel, or TenTussherPanfilov
im = ModifiedMS2v(dt, dtype=dtype)
im.u_gate = 0.1
im.u_crit = 0.1
im.tau_in = 0.15
im.tau_out = 1.5
im.tau_open = 105.0
im.tau_close = 185.0

case_name = "./Case_18"
mesh_dir = Path("./") / case_name

# 1. Initialise the Mondomain model
simulator = Monodomain(ionic_models=[im], T=simulation_time, dt=dt, dtype=dtype)
# 2. Load in the mesh files (.pts .elem .lon)
simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
# 3. Specify the conductivity for each region
simulator.add_conductivity(region_ids=[1, 2, 3, 4, 5, 6], il=0.3, it=0.06, el=None, et=None)
# 4. Specify the locations where stimulation is applied
simulator.add_stimulus(f"{mesh_dir}/{case_name}.vtx", 
                       start=0.0, 
                       duration=2.0, 
                       intensity=50)


# 5. Start the simulation
snapshot_interval = 1
Vm = simulator.solve(a_tol=1e-5,              # absolute tolerance
                     r_tol=1e-5,              # relative tolerance
                     max_iter=100,            # maximum number of iterations for each CG calculation
                     snapshot_interval=snapshot_interval,     # save the soluation after every 1 ms
                     verbose=True,
                     result_path="./Case_10_results")  # the folder in which the results are saved

# POSTPROCESSING: 
ATs = simulator.compute_activation_map(Vm=Vm, 
                                       snapshot_interval=snapshot_interval, 
                                       threshold=0)
print("ATs: ", ATs.min().item(), ATs.cpu().max().item(), flush=True)
RTs = simulator.compute_repolarization_map(Vm=Vm, 
                                           snapshot_interval=snapshot_interval, 
                                           threshold=-70)
print("RTs: ", RTs.min().item(), RTs.cpu().max().item(), flush=True)

simulator.vm_to_vtk(Vm=Vm, step=10)


