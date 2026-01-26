from typing import Dict, Optional
import torch
import numpy as np
from torchcor.core import MeshReader, MeshWriter, Matrices3D, Preconditioner, ConjugateGradient, Stimuli
from pathlib import Path
Tensor = torch.Tensor


    

class TorsoHeartMesh:
    def __init__(self):
        self.torso_nodes = None
        self.torso_elems = None
        self.torso_regions = None 
        self.torso_fibres = None

        self.old_to_new = None

        self.heart_mesh_dir = None

    def extract_mesh_on_tags(self, nodes, elems, regions, fibres, tags=[24, 25, 34, 35, 36]):
        tags = np.array(list(tags), dtype=np.int64)
        mask = np.isin(regions, tags)

        elems_keep = elems[mask]
        fibres_keep = fibres[mask]
        regions_keep = regions[mask]

        old_nodes_index = np.unique(elems_keep.reshape(-1))
        old_nodes_index_sorted = np.sort(old_nodes_index)

        nodes_extracted = nodes[old_nodes_index_sorted]

        self.old_to_new = np.full((nodes.shape[0],), -1, dtype=np.int64)
        self.old_to_new[old_nodes_index_sorted] = np.arange(old_nodes_index_sorted.size, dtype=np.int64)
        elems_extracted = self.old_to_new[elems_keep]
        
        return nodes_extracted, elems_extracted, regions_keep, fibres_keep

    def load_torsor_mesh(self, torsor_mesh_dir="/data/Bei/Torso/HC2/mesh", unit_conversion=1000):
        reader = MeshReader(torsor_mesh_dir)
        self.torso_nodes, self.torso_elems, self.torso_regions, self.torso_fibres = reader.read(unit_conversion=unit_conversion)
    
    def extract_heart_mesh(self, heart_mesh_dir="/data/Bei/Torso/HC2/heart", filename="1", tags=[24, 34, 36]):
        self.heart_mesh_dir = Path(heart_mesh_dir)
        self.heart_nodes, self.heart_elems, self.heart_regions, self.heart_fibres = self.extract_mesh_on_tags(self.torso_nodes, self.torso_elems, self.torso_regions, self.torso_fibres, tags)
        writer = MeshWriter(mesh_dir=self.heart_mesh_dir, filename=filename)

        writer.write(self.heart_nodes, self.heart_elems, self.heart_regions, self.heart_fibres)

    def load_stimulus_region(self, vtx_filepath):
        with Path(vtx_filepath).open("r") as f:
            region_size = int(f.readline().strip())
        region = np.loadtxt(vtx_filepath, dtype=int, skiprows=2)

        if len(region) != region_size:
            raise Exception(f"Error loading {vtx_filepath}")
        
        return region
    
    def save_stimulus_region(self, region, vtx_filepath):
        region = region.astype(np.int64).reshape(-1)
        with Path(vtx_filepath).open("w") as f:
            f.write(f"{len(region)}\n")
            f.write("extra\n")
            np.savetxt(f, region, fmt="%d")


    def convert_pacing_sites(self, pacing_sites_dir="/data/Bei/Torso/HC2/HPS"):
        pacing_sites_dir = Path(pacing_sites_dir)
        pacing_folder = self.heart_mesh_dir / "pacing" 
        pacing_folder.mkdir(exist_ok=True, parents=True)

        for filepath in pacing_sites_dir.iterdir():
            if filepath.suffix == ".vtx":
                region = self.load_stimulus_region(filepath)
                new_region = self.old_to_new[region]
                self.save_stimulus_region(new_region, pacing_folder / filepath.name)



class LeadField:
    def __init__(self, torso_nodes, torso_elems, sigma_t, heart_nodes, heart_elems, sigma_i, device, dtype):
        """
        torso_nodes: (N_torso, 3)
        torso_elems: (Ne_torso, 4)
        sigma_t: torso conductivity on torso elements, (Ne_torso,) or (Ne_torso, 3, 3)

        heart_nodes: (N_heart, 3)
        heart_elems: (Ne_heart, 4)
        sigma_i: intracellular conductivity tensor on heart elements, (Ne_heart, 3, 3)
        """
        self.device = device
        self.dtype = dtype

        self.torso_nodes = torso_nodes.to(self.device, self.dtype)
        self.torso_elems = torso_elems.to(self.device, torch.long)
        self.sigma_torso = sigma_t.to(self.device)  # keep its dtype as provided, but used in Matrices3D with dtype
        self.n_torso_nodes = int(self.torso_nodes.shape[0])

        self.heart_nodes = heart_nodes.to(self.device, self.dtype)
        self.heart_elems = heart_elems.to(self.device, torch.long)
        self.sigma_i = sigma_i.to(self.device, self.dtype)
        self.n_heart_nodes = int(self.heart_nodes.shape[0])

        self.electrodes: Dict[str, Tensor] = {}
        self.ground = "RL"

        self.K_torso: Optional[Tensor] = None  # sparse CSR
        self.K_heart: Optional[Tensor] = None      # sparse CSR

        self.projection_mode: Optional[str] = None
        self.heart_to_torso_node: Optional[Tensor] = None
        self.torso_tet_for_heart_node: Optional[Tensor] = None
        self.bary_w: Optional[Tensor] = None

        self.q_heart: Dict[str, Tensor] = {}

    def add_electrode(self, name, torso_nodes):
        idx = torch.tensor(list(torso_nodes), device=self.device, dtype=torch.long)
        self.electrodes[name] = idx

    def set_projection_direct_nodes(self, heart_to_torso_node: Tensor):
        """
        Direct mapping: phi_heart[i] = phi_torso[heart_to_torso_node[i]]
        """
        self.projection_mode = "direct"
        self.heart_to_torso_node = heart_to_torso_node.to(self.device, torch.long)

    def set_projection_barycentric(self, torso_tet_for_heart_node: Tensor, barycentric_weights: Tensor):
        """
        torso_tet_for_heart_node: (N_heart,) torso element index containing each heart node
        barycentric_weights: (N_heart, 4) weights for the four nodes of that torso tetrahedron
        """
        self.projection_mode = "barycentric"
        self.torso_tet_for_heart_node = torso_tet_for_heart_node.to(self.device, torch.long)
        self.bary_w = barycentric_weights.to(self.device, self.dtype)

    def build(self):
        """
        Assemble torso Laplace stiffness and heart intracellular stiffness.
        Call once.
        """
        torso_mats = Matrices3D(vertices=self.torso_nodes, tetrahedrons=self.torso_elems, device=self.device, dtype=self.dtype)
        K_torso, _ = torso_mats.assemble_matrices(self.sigma_torso)
        self.K_torso = K_torso.to_sparse_csr()

        heart_mats = Matrices3D(vertices=self.heart_nodes, tetrahedrons=self.heart_elems, device=self.device, dtype=self.dtype)
        K_heart, _ = heart_mats.assemble_matrices(self.sigma_i)
        self.K_heart = K_heart.to_sparse_csr()

    def project_phi_to_heart(self, phi_torso: Tensor) -> Tensor:
        if self.projection_mode == "direct":
            return phi_torso[self.heart_to_torso_node]

        if self.projection_mode == "barycentric":
            tet_nodes = self.torso_elems[self.torso_tet_for_heart_node]  # (N_heart, 4)
            phi_local = phi_torso[tet_nodes]                               # (N_heart, 4)
            return (phi_local * self.bary_w).sum(dim=1)
        
    def solve_dirichlet_penalty(
        self,
        fixed_nodes: Tensor,
        fixed_values: Tensor,
        alpha: Optional[float],
        a_tol: float,
        r_tol: float,
        max_iter: int,
        linear_guess: bool,
    ) -> Tensor:
        """
        Solve (K + alpha I_fixed) phi = alpha phi_fixed with CG.
        """
        K = self.K_torso
        n = self.n_torso_nodes

        fixed_nodes = fixed_nodes.to(self.device, torch.long)
        fixed_values = fixed_values.to(self.device, self.dtype)

        if alpha is None:
            diag = K.diagonal()
            scale = diag.abs().mean()
            alpha = float(scale.item()) * 1e8 + 1.0

        indices = torch.stack([fixed_nodes, fixed_nodes], dim=0)
        values = torch.full((fixed_nodes.numel(),), float(alpha), device=self.device, dtype=self.dtype)
        D = torch.sparse_coo_tensor(indices, values, size=(n, n), device=self.device, dtype=self.dtype).to_sparse_csr()
        A = K + D

        b = torch.zeros(n, device=self.device, dtype=self.dtype)
        b[fixed_nodes] = float(alpha) * fixed_values

        pcd = Preconditioner()
        pcd.create_Jocobi(A)

        cg = ConjugateGradient(pcd, A, dtype=torch.float64)
        cg.initialize(x=torch.zeros(n, device=self.device, dtype=self.dtype), linear_guess=linear_guess)

        phi, _ = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)

        return phi

    def precompute_electrode(
        self,
        name: str,
        phi_active: float = 1.0,
        phi_ground: float = 0.0,
        alpha: Optional[float] = None,
        a_tol: float = 1e-8,
        r_tol: float = 1e-8,
        max_iter: int = 20000,
        linear_guess: bool = True,
        demean_phi: bool = False,
    ) -> Tensor:
        """
        Precompute q vector for electrode 'name' relative to ground electrode.

        After this, unipolar signal is:
            U_name(t) = Vm(t) dot q_name
        """
        active_nodes = self.electrodes[name]
        ground_nodes = self.electrodes[self.ground]

        fixed_nodes = torch.cat([active_nodes, ground_nodes], dim=0)
        fixed_values = torch.cat(
            [
                torch.full((active_nodes.numel(),), float(phi_active), device=self.device, dtype=self.dtype),
                torch.full((ground_nodes.numel(),), float(phi_ground), device=self.device, dtype=self.dtype),
            ],
            dim=0,
        )

        phi_torso = self.solve_dirichlet_penalty(
            fixed_nodes=fixed_nodes,
            fixed_values=fixed_values,
            alpha=alpha,
            a_tol=a_tol,
            r_tol=r_tol,
            max_iter=max_iter,
            linear_guess=linear_guess,
        )

        if demean_phi:
            phi_torso = phi_torso - phi_torso.mean()

        phi_heart = self.project_phi_to_heart(phi_torso)
        q = self.K_heart @ phi_heart
        self.q_heart[name] = q

    def precompute_all(self):
        for name in self.electrodes.keys():
            if name != self.ground:
                self.precompute_electrode(name)

    def unipolar(self, Vm: Tensor, electrode: str) -> Tensor:
        return Vm @ self.q_heart[electrode]
    
    def read_electrodes(self, filepath, names=["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]):
        node_ids = np.loadtxt(filepath, dtype=np.int64, skiprows=1)

        electrodes = {
            name: np.array([node_id], dtype=np.int64)
            for name, node_id in zip(names, node_ids)
        }

        return electrodes

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



if __name__ == "__main__":
    thm = TorsoHeartMesh()
    thm.load_torsor_mesh(torsor_mesh_dir="/data/Bei/Torso/HC2/mesh", unit_conversion=1000)
    thm.extract_heart_mesh(heart_mesh_dir="/data/Bei/Torso/HC2/heart", filename="1", tags=[24, 25, 34, 35, 36])
    thm.convert_pacing_sites(pacing_sites_dir="/data/Bei/Torso/HC2/HPS")
    
    # lf = LeadField(torso_nodes, torso_elems, sigma_t, heart_nodes, heart_elems, sigma_i, device=torch.device("cuda:0"), dtype=torch.float64)

    # # Vm: (T, N_heart)
    # Vm = torch.load(mono.result_path / "Vm.pt").to(mono.device, mono.dtype)

    # lf = LeadField(
    #     torso_nodes=torso_nodes,
    #     torso_elems=torso_elems,
    #     sigma_t=sigma_torso_elem,
    #     heart_nodes=mono.nodes,
    #     heart_elems=mono.elems,
    #     sigma_i=mono.sigma_i,
    #     device=mono.device,
    #     dtype=mono.dtype,
    # )

    # lf.build()

    # # Projection: pick one
    # lf.set_projection_direct_nodes(heart_to_torso_node)
    # # or lf.set_projection_barycentric(torso_tet_for_heart_node=tet_id, barycentric_weights=w)

    # # Electrodes
    # lf.add_electrode("RA", ra_nodes)
    # lf.add_electrode("LA", la_nodes)
    # lf.add_electrode("LL", ll_nodes)
    # lf.add_electrode("RL", rl_nodes)
    # for k in ["V1","V2","V3","V4","V5","V6"]:
    #     lf.add_electrode(k, chest_nodes[k])

    # # Precompute per electrode lead field (once)
    # lf.precompute_all(a_tol=1e-10, r_tol=1e-10, max_iter=20000)

    # # Simulate ECG
    # ecg12 = lf.compute_12lead(Vm)