import torchcor as tc
from torchcor.electrophysiology import Monodomain
from torchcor.ionic import ModifiedMS2v, CourtemancheRamirezNattel
from pathlib import Path

tc.set_device("cuda:0")
simulation_time = 500
dt = 0.01

# ionic_model = ModifiedMS2v(dt)
# ionic_model.u_gate = 0.1
# ionic_model.u_crit = 0.1
# ionic_model.tau_in = 0.15
# ionic_model.tau_out = 1.5
# ionic_model.tau_open = 105.0
# ionic_model.tau_close = 185.0

ionic_model = CourtemancheRamirezNattel(dt)

home_dir = Path(f"{Path.home()}/Data/surface")
n_nodes_list = sorted([int(sub_dir.name) for sub_dir in home_dir.iterdir()])
print(n_nodes_list)

for n_nodes in n_nodes_list:
    mesh_dir = f"{home_dir}/{n_nodes}/"

    simulator = Monodomain(ionic_model, T=simulation_time, dt=dt)
    simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
    simulator.add_condutivity(region_ids=[0], il=0.4, it=0.4)

    simulator.add_stimulus(f"{mesh_dir}/{0}.vtx", 
                        start=0.0, 
                        duration=2.0, 
                        intensity=50)

    simulator.solve(a_tol=1e-5, 
                    r_tol=1e-5, 
                    max_iter=100, 
                    linear_guess=True,
                    plot_interval=10, 
                    verbose=True, 
                    format="")

