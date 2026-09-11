import torchcor as tc
from torchcor.electrophysiology import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

tc.set_device("cuda:0")
simulation_time = 500
dt = 0.01

home_dir = Path(f"{Path.home()}/Data/volume")

# n_nodes_list = sorted([int(sub_dir.name) for sub_dir in home_dir.iterdir()])
n_nodes_list = [3157665]
print(n_nodes_list)

for n_nodes in n_nodes_list:
    ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt)
    simulator = Monodomain(ionic_model, T=simulation_time, dt=dt)
    simulator.load_mesh(path=f"{home_dir}/{n_nodes}", unit_conversion=1000)
    simulator.add_condutivity([0], il=0.17, it=0.019, el=0.62, et=0.2)

    simulator.add_stimulus(f"{home_dir}/{n_nodes}/0.vtx", start=0.0, duration=2.0, intensity=50)

    simulator.solve(a_tol=1e-5, 
                    r_tol=1e-5, 
                    max_iter=100, 
                    linear_guess=True,
                    plot_interval=10, 
                    verbose=True, 
                    format="")


# TenTusscherPanfilov 14706 109.83 88.74 20.59 7017 36.2 7.26
# TenTusscherPanfilov 44831 110.5 89.05 20.96 6321 44.06 7.55
# TenTusscherPanfilov 102355 117.93 90.05 27.38 5720 62.1 8.18
# TenTusscherPanfilov 193151 137.16 98.04 38.61 5034 76.88 9.47
# TenTusscherPanfilov 325186 166.16 106.8 58.88 3898 95.84 10.69
# TenTusscherPanfilov 444870 220.88 142.24 78.15 3568 97.0 13.01
# TenTusscherPanfilov 620739 305.86 195.05 110.34 3419 98.76 15.81
# TenTusscherPanfilov 920028 476.37 293.28 182.58 3367 99.0 20.0
# TenTusscherPanfilov 1450392 849.76 504.42 344.85 3290 99.92 26.65
# TenTusscherPanfilov 2472555 1641.25 855.36 785.37 3660 100.0 39.99