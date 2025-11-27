import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v
from pathlib import Path

# Specify the GPU device to run the simulation on 
tc.set_device("cuda:0")
dtype = tc.float32
# The total simulation duration (ms)
simulation_time = 600
# time interval
dt = 0.01

# Load in the ionic model, such as MitchellSceaffer, CourtemancheRamirezNattel, or TenTussherPanfilov
ionic_model = ModifiedMS2v(dt, dtype=dtype)
ionic_model.u_gate = 0.1
ionic_model.u_crit = 0.1
ionic_model.tau_in = 0.15
ionic_model.tau_out = 1.5
ionic_model.tau_open = 105.0
ionic_model.tau_close = 185.0

case_name = "Case_10"
mesh_dir = Path("/app") / case_name

# 1. Initialise the Mondomain model
simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, dtype=dtype)
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
simulator.solve(a_tol=1e-5,              # absolute tolerance
                r_tol=1e-5,              # relative tolerance
                max_iter=100,            # maximum number of iterations for each CG calculation
                calculate_AT_RT=True,    # keep track of local activation time (LAT)
                linear_guess=True,
                snapshot_interval=1,     # save the soluation after every 1 ms
                verbose=True,
                result_path="./atrium")  # the folder in which the results are saved

simulator.pt_to_vtk()  # generate VTK files 
