import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

# Specify the GPU device for running the simulation
tc.set_device("cuda:0")
dtype = tc.float64
# The total simulation duration (ms)
simulation_time = 500
dt = 0.01

home_dir = Path.home()
mesh_dir = home_dir / "Data/ventricle/Case_1"
# Load in the ionic model. Here we use TenTussherPanfilov for the simulation on bi-ventricle
im = TenTusscherPanfilov(cell_type="ENDO", dt=dt, dtype=dtype)
# 1. Initialise the Mondomain model
simulator = Monodomain(ionic_models=[im], T=simulation_time, dt=dt, dtype=dtype)
# 2. Load in the mesh files (.pts .elem .lon)
simulator.load_mesh(path=mesh_dir)
# 3. Specify the conductivity for each region
simulator.add_conductivity([34, 35], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
simulator.add_conductivity([44, 45, 46], il=0.9074, it=0.3332, el=0.9074, et=0.3332)
# 4. Specify the locations where stimulation is applied
simulator.add_stimulus(mesh_dir / "LV_sf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "LV_pf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "LV_af.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "RV_sf.vtx", start=5.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

# 5. Start the simulation
snapshot_interval = 1
Vm = simulator.solve(a_tol=1e-5,              # absolute tolerance
                     r_tol=1e-5,              # relative tolerance
                     max_iter=100,            # maximum number of iterations for each CG calculation
                     snapshot_interval=snapshot_interval,     # save the soluation after every 1 ms
                     verbose=True,
                     result_path="./monodomain_biventricle")  # the folder in which the results are saved

# POSTPROCESSING: 
ATs = simulator.compute_activation_map(Vm=Vm, 
                                       snapshot_interval=snapshot_interval, 
                                       threshold=0)
print("ATs: ", ATs.min().item(), ATs.cpu().max().item(), flush=True)
RTs = simulator.compute_repolarization_map(Vm=Vm, 
                                           search_after=ATs,
                                           snapshot_interval=snapshot_interval, 
                                           threshold=-70)
print("RTs: ", RTs.min().item(), RTs.cpu().max().item(), flush=True)

# simulator.vm_to_vtk(Vm=Vm, step=10)
