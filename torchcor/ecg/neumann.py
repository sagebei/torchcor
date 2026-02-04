from typing import Dict, Optional
import torch
import numpy as np
from torchcor.core import MeshReader, MeshWriter, Matrices3D, Preconditioner, ConjugateGradient
from pathlib import Path
import matplotlib.pyplot as plt
from torchcor.core import Conductivity
Tensor = torch.Tensor
import warnings
warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state"
)


class LeadField:
    def __init__(self, torsor_mesh_dir,  heart_mesh_dir, device, dtype):
        self.device = device
        self.dtype = dtype

        self.load_torsor_mesh(torsor_mesh_dir)
        self.load_heart_mesh(heart_mesh_dir)

        self.torso_sigma = torch.zeros((self.torso_regions.shape[0], 3, 3), device=self.device, dtype=self.dtype)
        self.n_torso_nodes = int(self.torso_nodes.shape[0])

        self.heart_conductivity = Conductivity(self.heart_regions, dtype=self.dtype)
        self.heart_tags = []
        
        self.electrodes = {}
        self.ground = "RL"

        self.K_torso = None  
        self.K_heart = None  

        self.q_heart = {}
    
    def load_torsor_mesh(self, torsor_mesh_dir="/data/Bei/Torso/HC2/mesh", unit_conversion=1000):
        reader = MeshReader(torsor_mesh_dir)
        self.torso_nodes, self.torso_elems, self.torso_regions, _ = reader.read(unit_conversion=unit_conversion)
        self.torso_nodes = torch.from_numpy(self.torso_nodes).to(self.device, self.dtype)
        self.torso_elems = torch.from_numpy(self.torso_elems).to(self.device, torch.long)
        self.torso_regions = torch.from_numpy(self.torso_regions).to(self.device, torch.long)
    
    def load_heart_mesh(self, heart_mesh_dir="/data/Bei/Torso/HC2/heart", unit_conversion=1000):
        reader = MeshReader(heart_mesh_dir)
        self.heart_nodes, self.heart_elems, self.heart_regions, self.heart_fibres = reader.read(unit_conversion=unit_conversion)
        self.heart_nodes = torch.from_numpy(self.heart_nodes).to(self.device, self.dtype)
        self.heart_elems = torch.from_numpy(self.heart_elems).to(self.device, torch.long)
        self.heart_regions = torch.from_numpy(self.heart_regions).to(self.device, torch.long)
        self.heart_fibres = torch.from_numpy(self.heart_fibres).to(self.device, self.dtype)
    
    def add_torso_conductivity(self, tags, g):
        I = torch.eye(3, device=self.device, dtype=self.dtype)
        for tag in tags:
            mask = (self.torso_regions == int(tag))
            self.torso_sigma[mask] = float(g) * I

    def add_heart_conductivity(self, region_ids, il, it, el=None, et=None):
        self.heart_tags.extend(region_ids)
        self.heart_conductivity.add(region_ids, il, it, el, et)

    def build(self):
        """
        Assemble torso Laplace stiffness and heart intracellular stiffness.
        """
        self.sigma_i, self.sigma_e, _ = self.heart_conductivity.calculate_sigma(self.heart_fibres)

        torso_mats = Matrices3D(vertices=self.torso_nodes, tetrahedrons=self.torso_elems, device=self.device, dtype=self.dtype)
        heart_mask = torch.isin(self.torso_regions, torch.tensor(self.heart_tags, device=self.device, dtype=torch.long))
        
        self.torso_sigma[heart_mask] = self.sigma_i + self.sigma_e
        K_torso, _ = torso_mats.assemble_matrices(self.torso_sigma)
        self.K_torso = K_torso.to_sparse_csr()

        heart_mats = Matrices3D(vertices=self.heart_nodes, tetrahedrons=self.heart_elems, device=self.device, dtype=self.dtype)
        K_heart, _ = heart_mats.assemble_matrices(self.sigma_i)
        self.K_heart = K_heart.to_sparse_csr()

    def project_phi_to_heart(self, phi_torso: Tensor) -> Tensor:
        tags = torch.tensor(self.heart_tags, device=self.device, dtype=torch.long)
        mask = torch.isin(self.torso_regions, tags)

        elems_keep = self.torso_elems[mask]
        heart_to_torso_node = torch.unique(elems_keep.reshape(-1), sorted=True)
        
        return phi_torso[heart_to_torso_node]
    
    def default_alpha(self, K_csr: Tensor) -> float:
        """
        Choose a penalty scale from the mean absolute diagonal of K.
        """
        K_coo = K_csr.to_sparse_coo().coalesce()
        idx = K_coo.indices()
        val = K_coo.values()
        diag = val[idx[0] == idx[1]]
        scale = diag.abs().mean()
        return float(scale.item()) * 1e8 + 1.0

   

    def solve_neumann_penalty(self, source_nodes, source_values, a_tol, r_tol, max_iter):
        """
        Solve K phi = b with current sources and penalty grounding.
        """
        
        # Construct RHS from current sources
        b = torch.zeros(self.n_torso_nodes, device=self.device, dtype=self.dtype)
        b[source_nodes] = source_values
        
        # Apply penalty method to ground one node (removes singularity)
        # We choose the last node in source_nodes (assumed to be ground electrode)
        ground_idx = source_nodes[-1].item()
        alpha = self.default_alpha(self.K_torso)
        
        # Add penalty term: alpha * e_g * e_g^T (forces phi[ground_idx] ≈ 0)
        indices = torch.tensor([[ground_idx], [ground_idx]], device=self.device, dtype=torch.long)
        values = torch.tensor([alpha], device=self.device, dtype=self.dtype)
        D = torch.sparse_coo_tensor(
            indices, values,
            size=(self.n_torso_nodes, self.n_torso_nodes),
            device=self.device, dtype=self.dtype
        ).to_sparse_csr()
        
        A = self.K_torso + D
        
        # Solve with CG
        pcd = Preconditioner()
        pcd.create_Jocobi(A.to_sparse_coo())
        cg = ConjugateGradient(pcd, A, dtype=self.dtype)
        cg.initialize(
            x=torch.zeros(self.n_torso_nodes, device=self.device, dtype=self.dtype),
            linear_guess=False
        )
        phi, used_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        print(f"Neumann CG iterations: {used_iter}")
        
        if torch.isnan(phi).any():
            raise RuntimeError("Neumann solve diverged. Check conductivity values and mesh quality.")
        
        return phi

    def precompute_electrode_neumann(self, name, I, a_tol, r_tol, max_iter):
        source_nodes = torch.tensor([self.electrodes[name], self.electrodes[self.ground]], 
                                     device=self.device, 
                                     dtype=torch.long)
        source_values = torch.tensor([I, -I], device=self.device, dtype=self.dtype)
        
        phi_torso = self.solve_neumann_penalty(source_nodes, source_values, a_tol, r_tol, max_iter)
        
        # Remove mean (gauge freedom in Neumann problem)
        phi_torso = phi_torso - phi_torso.mean()
        
        # Project to heart mesh
        phi_heart = self.project_phi_to_heart(phi_torso)
        
        # Compute lead field vector: q = K_heart @ phi_heart
        q = self.K_heart @ phi_heart
        self.q_heart[name] = q

    def precompute_all_neumann(self, I=1.0, a_tol=1e-10, r_tol=1e-10, max_iter=20000):
        for name in self.electrodes.keys():
            if name != self.ground:
                print(f"Computing Neumann lead field for {name}")
                self.precompute_electrode_neumann(name, I, a_tol, r_tol, max_iter)


    def unipolar(self, Vm: Tensor, electrode: str) -> Tensor:
        """Compute unipolar signal for given electrode."""
        return Vm @ self.q_heart[electrode]
    
    def load_electrodes(self, filepath, names=["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]):
        node_ids = np.loadtxt(filepath, dtype=np.int64, skiprows=1).tolist()
        if len(node_ids) != 10:
            raise Exception(f"The number of electrodes in {filepath} is not ten.")
        self.electrodes = {
            name: node_id
            for name, node_id in zip(names, node_ids)
        }

    def compute_12lead(self, Vm: Tensor) -> Dict[str, Tensor]:
        Vm = Vm.to(self.device, self.dtype)
        required = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
        missing = [k for k in required if k not in self.electrodes]
        if missing:
            raise RuntimeError(f"Missing electrodes: {missing}")

        ra = self.unipolar(Vm, "RA")
        la = self.unipolar(Vm, "LA")
        ll = self.unipolar(Vm, "LL")

        lead_I = la - ra
        lead_II = ll - ra
        lead_III = ll - la

        aVR = ra - 0.5 * (la + ll)
        aVL = la - 0.5 * (ra + ll)
        aVF = ll - 0.5 * (ra + la)

        wct = (ra + la + ll) / 3.0

        V1 = self.unipolar(Vm, "V1") - wct
        V2 = self.unipolar(Vm, "V2") - wct
        V3 = self.unipolar(Vm, "V3") - wct
        V4 = self.unipolar(Vm, "V4") - wct
        V5 = self.unipolar(Vm, "V5") - wct
        V6 = self.unipolar(Vm, "V6") - wct

        return {
            "I": lead_I,
            "II": lead_II,
            "III": lead_III,
            "aVR": aVR,
            "aVL": aVL,
            "aVF": aVF,
            "V1": V1,
            "V2": V2,
            "V3": V3,
            "V4": V4,
            "V5": V5,
            "V6": V6,
        }
    
    def plot_ecg(self, ecg_dict, filename="ecg_12lead.png"):
        """
        Plot 12 lead ECG in a 3×4 grid using standard ordering.
        """
        order = [
            "I", "aVR", "V1", "V4", 
            "II", "aVL", "V2", "V5", 
            "III", "aVF", "V3", "V6",
        ]

        fig, axes = plt.subplots(3, 4, figsize=(12, 6), sharex=True)
        axes = axes.flatten()

        for ax, lead in zip(axes, order):
            sig = ecg_dict[lead].detach().cpu().numpy()
            ax.plot(sig, lw=1.2)
            ax.set_title(lead, fontsize=10)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.supylabel("Potential (a.u.)", fontsize=11)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    torsor_mesh_dir = "/data/Bei/Torso/HC2/mesh"
    heart_mesh_dir = "/data/Bei/Torso/HC2/heart"
    
    # Initialize lead field solver
    lf = LeadField(torsor_mesh_dir, heart_mesh_dir, 
                   device=torch.device("cuda:0"), dtype=torch.float32)
    
    # Set torso conductivities (S/m)
    lf.add_torso_conductivity([20, 21], g=0.25)
    lf.add_torso_conductivity([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23], g=0.6667)
    lf.add_torso_conductivity([3], g=0.05)
    lf.add_torso_conductivity([1, 7], g=0.2472)
    lf.add_torso_conductivity([4], g=0.1667)
    lf.add_torso_conductivity([5], g=0.1667)
    lf.add_torso_conductivity([6], g=0.0714)
    lf.add_torso_conductivity([2], g=0.117)
    lf.add_torso_conductivity([8], g=0.1)
    lf.add_torso_conductivity([9], g=0.1)

    # Set heart conductivities
    lf.add_heart_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
    lf.add_heart_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

    # Build stiffness matrices
    lf.build()
    
    # Load electrodes
    lf.load_electrodes("/data/Bei/Torso/HC3/electrodes/lf_src.vtx")

    # Load transmembrane potential
    Vm = torch.load("./biventricle/Vm.pt").to(lf.device, lf.dtype)
    
    lf.precompute_all_neumann(I=1.0, a_tol=1e-8, r_tol=1e-8, max_iter=10000)
    ecg12_neumann = lf.compute_12lead(Vm)
    lf.plot_ecg(ecg12_neumann, filename="ecg_neumann.png")
    
