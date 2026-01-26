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

mesh_dir = Path("/data/Bei/Torso/HC2/heart")
# Load in the ionic model. Here we use TenTussherPanfilov for the simulation on bi-ventricle
ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt, dtype=dtype)
# ionic_model = ModifiedMS2v(dt=dt, dtype=dtype)
# 1. Initialise the Mondomain model
simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, dtype=dtype)
# 2. Load in the mesh files (.pts .elem .lon)
simulator.load_mesh(path=mesh_dir)
# 3. Specify the conductivity for each region
simulator.add_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
simulator.add_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)
# 4. Specify the locations where stimulation is applied
simulator.add_stimulus(mesh_dir / "pacing" / "LV_sf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "LV_pf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "LV_af.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "RV_sf.vtx", start=5.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

# 5. Start the simulation
simulator.solve(a_tol=1e-5,                  # absolute tolerance in CG
                r_tol=1e-5,                  # relative tolerance
                max_iter=1000,                # maximum number of iterations for each CG calculation
                calculate_AT_RT=True,        # keep track of local activation time (LAT)
                linear_guess=True,
                snapshot_interval=10,         # save the soluation after every 1 ms
                verbose=True,
                result_path="./biventricle") # to folder in which the results are saved
# simulator.pt_to_vtk()