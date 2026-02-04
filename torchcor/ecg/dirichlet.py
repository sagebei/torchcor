from typing import Dict
import torch
import numpy as np
from torchcor.core import MeshReader, Matrices3D, Preconditioner, ConjugateGradient
from pathlib import Path
import matplotlib.pyplot as plt
from torchcor.core import Conductivity
import warnings
warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state"
)
Tensor = torch.Tensor


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
        mask = torch.isin(self.torso_regions, torch.tensor(self.heart_tags, device=self.device, dtype=torch.long))
        
        self.torso_sigma[mask] = self.sigma_i + self.sigma_e
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
         
        
    def solve_dirichlet_penalty(self, fixed_nodes, fixed_values, a_tol, r_tol, max_iter):
        """
        Solve (K + alpha I_fixed) phi = alpha phi_fixed with CG.
        """
        
        alpha = self.default_alpha(self.K_torso)

        indices = torch.stack([fixed_nodes, fixed_nodes], dim=0)
        values = torch.full((fixed_nodes.numel(),), float(alpha), device=self.device, dtype=self.dtype)
        D = torch.sparse_coo_tensor(indices, values, size=(self.n_torso_nodes, self.n_torso_nodes), device=self.device, dtype=self.dtype).to_sparse_csr()
        A = self.K_torso + D

        b = torch.zeros(self.n_torso_nodes, device=self.device, dtype=self.dtype)
        b[fixed_nodes] = float(alpha) * fixed_values

        pcd = Preconditioner()
        pcd.create_Jocobi(A.to_sparse_coo())
        cg = ConjugateGradient(pcd, A, dtype=torch.float64)
        cg.initialize(x=torch.zeros(self.n_torso_nodes, device=self.device, dtype=self.dtype), linear_guess=False)
        phi, used_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        print(used_iter)

        return phi

    def precompute_electrode(self, name, a_tol, r_tol, max_iter):
        """
        Precompute q vector for electrode 'name' relative to ground electrode.

        After this, unipolar signal is:
            U_name(t) = Vm(t) dot q_name
        """
        fixed_nodes = torch.tensor([self.electrodes[name], 
                                    self.electrodes[self.ground]], 
                                    device=self.device, 
                                    dtype=torch.long)
        fixed_values = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)

        phi_torso = self.solve_dirichlet_penalty(fixed_nodes, fixed_values, a_tol, r_tol, max_iter)
        
        if torch.isnan(phi_torso).any():
            raise RuntimeError("phi_torso contains NaNs. Solver diverged or matrix is singular.")

        phi_torso = phi_torso - phi_torso.mean()

        phi_heart = self.project_phi_to_heart(phi_torso)
        q = self.K_heart @ phi_heart
        self.q_heart[name] = q

    def precompute_all(self, a_tol=1e-10, r_tol=1e-10, max_iter=20000):
        for name in self.electrodes.keys():
            print(name)
            if name != self.ground:
                self.precompute_electrode(name, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)

    def unipolar(self, Vm: Tensor, electrode: str) -> Tensor:
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
        """
        Standard 12 lead ECG construction.
        Requires electrodes: RA, LA, LL, RL, V1..V6 and ground set to RL.
        """
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
    

    def plot_ecg(self, ecg_dict, filename="ecg_dirichlet.png"):
        """
        Plot 12-lead ECG in a 3×4 grid using standard ordering.
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

        # Common labels
        fig.supylabel("Potential (a.u.)", fontsize=11)

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=300)
        plt.close(fig)


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




if __name__ == "__main__":
    torsor_mesh_dir = "/data/Bei/Torso/HC2/mesh"
    heart_mesh_dir = "/data/Bei/Torso/HC2/heart"
    
    lf = LeadField(torsor_mesh_dir,  heart_mesh_dir, device=torch.device("cuda:0"), dtype=torch.float32)
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

    lf.add_heart_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
    lf.add_heart_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

    # # Vm: (T, N_heart)
    Vm = torch.load("./biventricle/Vm.pt").to(lf.device, lf.dtype)
    
    
    lf.build()

    lf.load_electrodes("/data/Bei/Torso/HC3/electrodes/lf_src.vtx")

    # # Precompute per electrode lead field (once)
    lf.precompute_all(a_tol=1e-8, r_tol=1e-8, max_iter=10000)

    # # Simulate ECG
    ecg12 = lf.compute_12lead(Vm)
    lf.plot_ecg(ecg12)