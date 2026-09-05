<p align="center">
  <img src="docs/logo.png" alt="TorchCor logo" width="160"/>
</p>

<p align="center">
  <b>GPU-accelerated cardiac electrophysiology simulation in PyTorch</b>
</p>

<p align="center">
 <a href="https://torchcor.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-readthedocs-blue?logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img src="https://img.shields.io/badge/python-3.10+-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia">
</p>



**TorchCor** is a high-performance simulator for cardiac electrophysiology (CEP) using the finite element method (FEM) on general-purpose GPUs. Built on top of PyTorch, TorchCor delivers substantial computational acceleration for large-scale CEP simulations, with seamless integration into modern deep learning workflows and efficient handling of complex mesh geometries.

**TorchCor offers:**

- 🚀 Fast, scalable CEP simulations on large and complex heart meshes  
- 🔗 Seamless integration with PyTorch and scientific machine learning workflows  
- ⚙️ Support for a wide range of ionic models and conductivity heterogeneity  
- 📈 Generation of precise local activation and repolarization time maps with monodomain or reaction-eikonal
- 🩺 Simulation of clinically relevant **12-lead ECG** signals via the lead-field method


## 🫀 Simulation Previews
Below are simulation results showcasing the electrical activation patterns over time in the left atrium and bi-ventricle.

<table>
  <tr>
    <td align="center">
      <img src="docs/atrium.gif" alt="Left Atrium simulation" width="300"/><br/>
      <strong>3D left atrium surface mesh</strong>
    </td>
    <td align="center">
      <img src="docs/biv.gif" alt="Bi-ventricle simulation" width="300"/><br/>
      <strong>3D bi-ventricle volume mesh</strong>
    </td>
  </tr>
</table>

## 📦 Installation

```bash
pip install torchcor
```
> **Note:** Requires PyTorch with CUDA support for GPU acceleration.

## 🚀 Quickstart Example

Here’s a concise example to run a simulation using the **TenTusscher-Panfilov** ionic model on a bi-ventricle mesh.

🧩 **Mesh file:**  
You can download the bi-ventricle mesh from the following link:  
[Google Drive – Bi-ventricle Mesh](https://drive.google.com/drive/folders/1_lWTNaTni4GElSSS8tdc8YmbwOZ4PcaK?usp=drive_link)

The inputs are:

- `.pts` file: coordinates of the vertices
- `.elem` file: connectivities of the mesh
- `.lon` file: fibre orientations
- One or more `.vtx` files: pacing locations

Running the following code will produce:  
- a list of membrane potentials, each saved after every 1 ms of the simulation
- a local activation time map and a repolarisation time map  

all saved in `.pt` file format readable by `torch.load`. This execution takes 73 seconds on RTX 5090. 

```python
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
# Load in the ionic model. Here we use TenTusscherPanfilov for the simulation on bi-ventricle
im = TenTusscherPanfilov(cell_type="ENDO", dt=dt, dtype=dtype)
# 1. Initialise the Monodomain model
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
                     snapshot_interval=snapshot_interval,     # save the solution after every 1 ms
                     verbose=True,
                     result_path="./biventricle")  # the folder in which the results are saved

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
```

## 🩺 12-Lead ECG via the Lead-Field Method

TorchCor reconstructs the clinical **12-lead ECG** directly from a cardiac simulation using the **lead-field (reciprocity) method** on a coupled heart–torso mesh. Each electrode's lead field is precomputed with a single elliptic solve; the entire ECG time series then follows from one matrix–vector product with the transmembrane potential `Vm`, so evaluating the ECG over a whole beat is effectively free.

<p align="center">
  <img src="docs/ecg_leadfield.png" alt="TorchCor simulated 12-lead ECG" width="720"/><br/>
  <em>Simulated 12-lead ECG from a full heart–torso model, reconstructed with the lead-field reciprocity method.</em>
</p>

```python
import torch
from torchcor.ecg import LeadField

# Coupled heart–torso mesh; precompute one lead field per electrode
lf = LeadField(torso_mesh_dir, heart_mesh_dir,
               device=torch.device("cuda:0"), dtype=torch.float64)

# Passive torso/organ conductivities (S/m) + anisotropic bidomain heart conductivities
lf.add_torso_conductivity([10, 11, 12], g=0.6667)
lf.add_heart_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
lf.build()

# Electrodes (V1–V6, RA, LA, RL, LL); one elliptic solve per lead
lf.load_electrodes("electrodes/lf_src.vtx")
lf.precompute_all()

# Vm: (T, N_heart) from the EP simulation  ->  dict of the 12 standard leads
ecg = lf.compute_12lead(Vm)
```

A complete heart → torso → 12-lead pipeline is provided in [`torchcor/demo/ecg_12lead.py`](torchcor/demo/ecg_12lead.py).

## 🐳 Docker Support

For a GPU-enabled Docker setup to run TorchCor without installing dependencies on your host system, please see the `docker/` folder.  

## 📖 Citation

If you use TorchCor in academic work, please consider citing:

```bibtex
@article{zhou2026torchcor,
  title={TorchCor: High-performance cardiac electrophysiology simulations with the finite element method on GPUs},
  author={Zhou, Bei and Balmus, Maximilian and Corrado, Cesare and Cicci, Ludovica and Qian, Shuang and Niederer, Steven A},
  journal={SoftwareX},
  volume={33},
  pages={102521},
  year={2026},
  publisher={Elsevier}
}
```

## 👩‍💻 Contributors

**TorchCor** is developed and maintained by [Bei Zhou](mailto:bei.zhou@imperial.ac.uk), [Maximilian Balmus](mailto:mbalmus@turing.ac.uk), [Cesare Corrado](mailto:c.corrado@imperial.ac.uk), [Shuang Qian](mailto:s.qian23@imperial.ac.uk), and [Steven A. Niederer​](mailto:s.niederer@imperial.ac.uk) in the [Cardiac Electro-Mechanics Research Group (CEMRG)](https://www.cemrg.co.uk/) at Imperial College London.

We welcome contributions from the community! Feel free to open issues or submit pull requests.
