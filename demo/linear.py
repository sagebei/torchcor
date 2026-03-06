import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import CourtemancheRamirezNattel
from pathlib import Path

tc.set_device("cuda:0")
dtype = tc.float64
simulation_time = 6000
dt = 0.02

im1 = CourtemancheRamirezNattel(dt, region_ids=[0], dtype=dtype)
im1.GNa *= 1
im1.Gto *= 1
im1.GKr *= 1.
im1.GCaL *= 1
im1.GK1 *= 1
im1.factorGKur *= 1.

im2 = CourtemancheRamirezNattel(dt, region_ids=[10], dtype=dtype)
im2.GNa *= 1
im2.Gto *= 0.334
im2.GKr *= 1.
im2.GCaL *= 0.5
im2.GK1 *= 2
im2.factorGKur *= 1.

im3 = CourtemancheRamirezNattel(dt, region_ids=[11, 12, 14], dtype=dtype)
im3.GNa *= 1
im3.Gto *= 0.334
im3.GKr *= 1.
im3.GCaL *= 0.5
im3.GK1 *= 2
im3.factorGKur *= 1.

im4 = CourtemancheRamirezNattel(dt, region_ids=[13], dtype=dtype)
im4.GNa *= 1
im4.Gto *= 0.334
im4.GKr *= 1.
im4.GCaL *= 0.32
im4.GK1 *= 2
im4.factorGKur *= 1.

im5 = CourtemancheRamirezNattel(dt, region_ids=[21, 22, 23, 24, 25, 26, 27, 28], dtype=dtype)
im5.GNa *= 1
im5.Gto *= 0.334
im5.GKr *= 1.
im5.GCaL *= 0.22
im5.GK1 *= 2
im5.factorGKur *= 1.


mesh_dir = Path("/home/bzhou6/Data/Mesh_12928433") 

# im2, im3, im4, im5
simulator = Monodomain(ionic_models=[im1, im2, im3, im4, im5], T=simulation_time, dt=dt, dtype=dtype)

simulator.load_mesh(path=mesh_dir, unit_conversion=1000)

simulator.add_conductivity(region_ids=[0], il=0.3, it=0.3, el=None, et=None)
simulator.add_conductivity(region_ids=[10], il=0.702, it=0.181, el=None, et=None)
simulator.add_conductivity(region_ids=[11, 14, 23, 27, 18], il=0.4, it=0.11, el=None, et=None)
simulator.add_conductivity(region_ids=[12, 13, 21, 22, 24, 25, 26], il=0.4, it=0.107, el=None, et=None)
simulator.add_conductivity(region_ids=[32], il=50, it=50, el=None, et=None)

simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=0.0, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=185, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=370, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=530, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=675, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=820, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=955, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1090, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1225, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1355, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1485, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1610, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1735, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1860, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=1980, duration=2.0, intensity=20)
simulator.add_stimulus(f"{mesh_dir}/LA_Pst_Sept.vtx", start=2095, duration=2.0, intensity=20)

snapshot_interval = 5
Vm = simulator.solve(a_tol=1e-4,              
                     r_tol=1e-4,              
                     max_iter=100,            
                     snapshot_interval=snapshot_interval,    
                     verbose=True,
                     result_path="./linear")  


# POSTPROCESSING: 
ATs = simulator.compute_activation_map(Vm=Vm, 
                                       snapshot_interval=snapshot_interval, 
                                       threshold=-10)
print("ATs: ", ATs.min().item(), ATs.cpu().max().item(), flush=True)
RTs = simulator.compute_repolarization_map(Vm=Vm, 
                                           snapshot_interval=snapshot_interval, 
                                           threshold=-70,
                                           first=True)
print("Frist RTs: ", RTs.min().item(), RTs.cpu().max().item(), flush=True)


RTs = simulator.compute_repolarization_map(Vm=Vm, 
                                           snapshot_interval=snapshot_interval, 
                                           threshold=-70,
                                           first=False)
print("Last RTs: ", RTs.min().item(), RTs.cpu().max().item(), flush=True)

simulator.vm_to_vtk(Vm=Vm, step=2)
simulator.vm_to_igb(Vm=Vm)

