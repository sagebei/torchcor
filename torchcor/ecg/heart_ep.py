import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

tc.set_device("cuda:1")
dtype = tc.float32
simulation_time = 500
dt = 0.01

mesh_dir = Path("/data/Bei/Torso/HC2/heart")
ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt, dtype=dtype)

simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, dtype=dtype)
simulator.load_mesh(path=mesh_dir)
simulator.add_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
simulator.add_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

simulator.add_stimulus(mesh_dir / "pacing" / "LV_sf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "LV_pf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "LV_af.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "RV_sf.vtx", start=5.0, duration=1.0, intensity=100)
simulator.add_stimulus(mesh_dir / "pacing" / "RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

simulator.solve(a_tol=1e-5,                 
                r_tol=1e-5,                 
                max_iter=1000,               
                calculate_AT_RT=True,       
                linear_guess=True,
                snapshot_interval=1,         
                verbose=True,
                result_path="./biventricle")

# simulator.pt_to_vtk()
