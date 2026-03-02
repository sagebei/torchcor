"""
ECG Forward Problem Solver — Clean Rewrite
===========================================

Two methods implemented for cross-validation:

  A) Lead Field (Reciprocal) method  — Potse, Front. Physiol. 2018;9:370
  B) Full Forward Solve              — Bishop & Plank, IEEE TBME 2011

Both use the same FEM discretisation and must produce identical ECGs.

Theory
------
Bidomain equation for the extracellular potential phi_e:

    div( (sigma_i + sigma_e) grad(phi_e) ) = -div( sigma_i grad(Vm) )   in Omega
    (sigma_i + sigma_e) grad(phi_e) . n    = 0                          on dOmega

FEM weak form (test function w, integration by parts on both sides):

    w^T  K_{ie}  phi_e  =  -w^T  K_i  Vm

where K_{ie} is assembled with (sigma_i + sigma_e) on heart elements and
sigma_T on torso elements, and K_i is assembled with sigma_i on heart
elements only, embedded into the torso node numbering.

Lead field method (reciprocal):
    Solve  K_{ie} Z = f_electrode   once per lead
    Then   V_lead(t) = Vm(t)^T  K_i^{heart}  Z_heart   for each time step

Full forward method:
    Solve  K_{ie} phi_e = -K_i^{torso} Vm(t)   for each time step
    Then   V_lead(t) = c^T phi_e   (electrode linear combination)

References
----------
[1] Potse M. Scalable and accurate ECG simulation for reaction-diffusion
    models of the human heart. Front Physiol. 2018;9:370.
[2] Geselowitz DB. On the theory of the electrocardiogram. Proc IEEE.
    1989;77:857-876.
[3] Bishop MJ, Plank G. Bidomain ECG simulations using an augmented
    monodomain model for the cardiac source. IEEE TBME. 2011;58:2297-2307.
"""

from typing import Dict, Optional, Tuple
import torch
import numpy as np
from torchcor.core import (
    MeshReader,
    Matrices3D,
    Preconditioner,
    ConjugateGradient,
    Conductivity,
)
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

Tensor = torch.Tensor


# =====================================================================
#  Utility: robust CG solve with penalty grounding
# =====================================================================

def solve_singular_system(
    K_csr: Tensor,
    b: Tensor,
    ground_node: int,
    device: torch.device,
    dtype: torch.dtype,
    a_tol: float = 1e-10,
    r_tol: float = 1e-10,
    max_iter: int = 20000,
    penalty_factor: float = 1e6,
) -> Tensor:
    """
    Solve K x = b where K is singular (pure Neumann).

    Uses the penalty method: augment K with alpha * e_g e_g^T to pin
    x[ground_node] ~ 0, removing the one-dimensional null space.

    Parameters
    ----------
    K_csr : sparse CSR tensor, (N, N)
    b     : dense tensor, (N,)
    ground_node : index of the node to pin to zero
    penalty_factor : multiplier on mean(|diag(K)|); 1e6 is safe
    """
    N = b.shape[0]

    # Compute penalty magnitude from diagonal of K
    K_coo = K_csr.to_sparse_coo().coalesce()
    idx = K_coo.indices()
    val = K_coo.values()
    diag_mask = idx[0] == idx[1]
    diag_abs_mean = val[diag_mask].abs().mean().item()
    alpha = diag_abs_mean * penalty_factor

    # Build rank-1 penalty: alpha * e_g e_g^T
    penalty = torch.sparse_coo_tensor(
        torch.tensor([[ground_node], [ground_node]], device=device, dtype=torch.long),
        torch.tensor([alpha], device=device, dtype=dtype),
        size=(N, N),
        device=device,
        dtype=dtype,
    ).to_sparse_csr()

    A = K_csr + penalty

    # Preconditioned CG
    pcd = Preconditioner()
    pcd.create_Jocobi(A.to_sparse_coo())
    cg = ConjugateGradient(pcd, A, dtype=dtype)
    cg.initialize(
        x=torch.zeros(N, device=device, dtype=dtype),
        linear_guess=False,
    )
    x, iters = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)

    if torch.isnan(x).any():
        raise RuntimeError(
            f"CG diverged after {iters} iterations. "
            "Check conductivity values and mesh quality."
        )

    return x


# =====================================================================
#  Core class
# =====================================================================

class ECGSolver:
    """
    Compute 12-lead ECGs from transmembrane potentials using the
    lead field (reciprocal) method or the full forward solve.

    The heart mesh MUST be a subset of the torso mesh, extracted via
    TorsoHeartMesh (or equivalent). The node correspondence is
    established by the heart_to_torso_node mapping.
    """

    def __init__(
        self,
        torso_mesh_dir: str,
        heart_mesh_dir: str,
        heart_to_torso_node: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Parameters
        ----------
        torso_mesh_dir : path to torso mesh files
        heart_mesh_dir : path to heart mesh files (extracted subset)
        heart_to_torso_node : array of shape (n_heart_nodes,)
            heart_to_torso_node[j] = torso node index for heart node j.
            Built by TorsoHeartMesh.extract_mesh_on_tags().
        device : torch device
        dtype  : torch dtype (float32 or float64)
        """
        self.device = device
        self.dtype = dtype

        # Load meshes
        self._load_torso_mesh(torso_mesh_dir)
        self._load_heart_mesh(heart_mesh_dir)

        # Store node mapping
        self.heart_to_torso = torch.from_numpy(
            heart_to_torso_node.astype(np.int64)
        ).to(device)

        # Validate dimensions
        assert self.heart_to_torso.shape[0] == self.heart_nodes.shape[0], (
            f"Mapping has {self.heart_to_torso.shape[0]} entries but heart "
            f"mesh has {self.heart_nodes.shape[0]} nodes."
        )

        self.n_torso = int(self.torso_nodes.shape[0])
        self.n_heart = int(self.heart_nodes.shape[0])

        # Conductivity storage
        self.torso_sigma = torch.zeros(
            (self.torso_regions.shape[0], 3, 3), device=device, dtype=dtype
        )
        self.heart_conductivity = Conductivity(self.heart_regions, dtype=dtype)
        self.heart_tags: list = []

        # Assembled matrices (set by build())
        self.K_torso: Optional[Tensor] = None   # (sigma_i+sigma_e) on heart, sigma_T elsewhere
        self.K_heart: Optional[Tensor] = None   # sigma_i on heart mesh
        self.K_i_torso: Optional[Tensor] = None  # sigma_i embedded in torso node numbering

        # Electrodes
        self.electrodes: Dict[str, int] = {}
        self.ground: str = "RL"

        # Lead field vectors (filled by precompute)
        self.lead_fields: Dict[str, Tensor] = {}   # Z_heart per electrode
        self.q_vectors: Dict[str, Tensor] = {}      # K_heart @ Z_heart per electrode

    # -----------------------------------------------------------------
    # Mesh I/O
    # -----------------------------------------------------------------

    def _load_torso_mesh(self, mesh_dir, unit_conversion=1000):
        reader = MeshReader(mesh_dir)
        n, e, r, _ = reader.read(unit_conversion=unit_conversion)
        self.torso_nodes = torch.from_numpy(n).to(self.device, self.dtype)
        self.torso_elems = torch.from_numpy(e).to(self.device, torch.long)
        self.torso_regions = torch.from_numpy(r).to(self.device, torch.long)

    def _load_heart_mesh(self, mesh_dir, unit_conversion=1000):
        reader = MeshReader(mesh_dir)
        n, e, r, f = reader.read(unit_conversion=unit_conversion)
        self.heart_nodes = torch.from_numpy(n).to(self.device, self.dtype)
        self.heart_elems = torch.from_numpy(e).to(self.device, torch.long)
        self.heart_regions = torch.from_numpy(r).to(self.device, torch.long)
        self.heart_fibres = torch.from_numpy(f).to(self.device, self.dtype)

    # -----------------------------------------------------------------
    # Conductivity assignment
    # -----------------------------------------------------------------

    def set_torso_conductivity(self, tags, g: float):
        """Assign isotropic conductivity g (S/m) to torso regions with given tags."""
        I3 = torch.eye(3, device=self.device, dtype=self.dtype)
        for tag in tags:
            mask = self.torso_regions == int(tag)
            self.torso_sigma[mask] = g * I3

    def set_heart_conductivity(self, region_ids, il, it, el=None, et=None):
        """
        Set anisotropic heart conductivities.

        il, it : intracellular longitudinal / transverse (S/m)
        el, et : extracellular longitudinal / transverse (S/m)
        """
        self.heart_tags.extend(region_ids)
        self.heart_conductivity.add(region_ids, il, it, el, et)

    # -----------------------------------------------------------------
    # Assembly
    # -----------------------------------------------------------------

    def build(self):
        """
        Assemble three stiffness matrices:

        1. K_torso: full torso stiffness with (sigma_i + sigma_e) on heart
           elements and the assigned sigma_T elsewhere.
           This is the operator for both the lead field PDE and the
           forward problem left hand side.

        2. K_heart: intracellular stiffness on the heart mesh (heart
           node numbering). Used for the lead field inner product.

        3. K_i_torso: intracellular stiffness assembled on the torso mesh
           (torso node numbering), with sigma_i on heart elements and 0
           elsewhere. Used as the source operator in the full forward solve.
        """
        # Compute heart sigma_i, sigma_e from fibre directions
        self.sigma_i, self.sigma_e, _ = (
            self.heart_conductivity.calculate_sigma(self.heart_fibres)
        )

        # Heart tags as tensor for masking
        htags = torch.tensor(self.heart_tags, device=self.device, dtype=torch.long)
        heart_mask = torch.isin(self.torso_regions, htags)

        # ----- K_torso: (sigma_i + sigma_e) on heart, sigma_T elsewhere -----
        self.torso_sigma[heart_mask] = self.sigma_i + self.sigma_e
        torso_mats = Matrices3D(
            vertices=self.torso_nodes,
            tetrahedrons=self.torso_elems,
            device=self.device,
            dtype=self.dtype,
        )
        K, _ = torso_mats.assemble_matrices(self.torso_sigma)
        self.K_torso = K.to_sparse_csr()

        # ----- K_heart: sigma_i on heart mesh (heart numbering) -----
        heart_mats = Matrices3D(
            vertices=self.heart_nodes,
            tetrahedrons=self.heart_elems,
            device=self.device,
            dtype=self.dtype,
        )
        Kh, _ = heart_mats.assemble_matrices(self.sigma_i)
        self.K_heart = Kh.to_sparse_csr()

        # ----- K_i_torso: sigma_i embedded in torso numbering -----
        # Zero conductivity for non-heart elements, sigma_i for heart
        sigma_i_full = torch.zeros_like(self.torso_sigma)
        sigma_i_full[heart_mask] = self.sigma_i
        Ki_full, _ = torso_mats.assemble_matrices(sigma_i_full)
        self.K_i_torso = Ki_full.to_sparse_csr()

        print(
            f"Assembly complete: "
            f"torso {self.n_torso} nodes, "
            f"heart {self.n_heart} nodes, "
            f"heart elements {heart_mask.sum().item()}"
        )

    # -----------------------------------------------------------------
    # Electrode loading
    # -----------------------------------------------------------------

    def load_electrodes(
        self,
        filepath: str,
        names=None,
    ):
        if names is None:
            names = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
        node_ids = np.loadtxt(filepath, dtype=np.int64, skiprows=1).tolist()
        if len(node_ids) != len(names):
            raise ValueError(
                f"Expected {len(names)} electrodes, got {len(node_ids)}"
            )
        self.electrodes = dict(zip(names, node_ids))
        print(f"Loaded {len(self.electrodes)} electrodes, ground = {self.ground}")

    # -----------------------------------------------------------------
    # METHOD A: Lead Field (Reciprocal)
    # -----------------------------------------------------------------

    def _solve_lead_field(
        self, electrode_name: str, I: float,
        a_tol: float, r_tol: float, max_iter: int,
    ):
        """
        Solve the reciprocal problem for one electrode:

            K_torso  Z  =  I * delta(x_electrode) - I * delta(x_ground)

        with penalty grounding at the RL electrode.
        Z is the lead field potential over the full torso.
        """
        # Build RHS: unit current in at electrode, out at ground
        b = torch.zeros(self.n_torso, device=self.device, dtype=self.dtype)
        b[self.electrodes[electrode_name]] = I
        b[self.electrodes[self.ground]] = -I

        # Solve with penalty at ground node
        Z_torso = solve_singular_system(
            self.K_torso, b,
            ground_node=self.electrodes[self.ground],
            device=self.device, dtype=self.dtype,
            a_tol=a_tol, r_tol=r_tol, max_iter=max_iter,
        )

        # NO mean subtraction. The penalty already fixes the gauge.

        # Extract Z at heart nodes
        Z_heart = Z_torso[self.heart_to_torso]

        # Lead field vector: q = K_i^{heart} @ Z_heart
        # At runtime: V(t) = Vm(t) @ q
        q = self.K_heart @ Z_heart

        self.lead_fields[electrode_name] = Z_heart
        self.q_vectors[electrode_name] = q

    def precompute_lead_fields(
        self,
        I: float = 1.0,
        a_tol: float = 1e-10,
        r_tol: float = 1e-10,
        max_iter: int = 20000,
    ):
        """Compute lead field vectors for all electrodes except the ground."""
        for name in self.electrodes:
            if name != self.ground:
                print(f"  Lead field: {name} ... ", end="", flush=True)
                self._solve_lead_field(name, I, a_tol, r_tol, max_iter)
                print("done")

    def ecg_lead_field(self, Vm: Tensor, electrode: str) -> Tensor:
        """
        Compute unipolar ECG signal using the lead field method.

        Vm : (n_time, n_heart_nodes)
        returns : (n_time,)

        V(t) = Vm(t)^T  K_i^{heart}  Z_heart  =  Vm @ q
        """
        return Vm @ self.q_vectors[electrode]

    # -----------------------------------------------------------------
    # METHOD B: Full Forward Solve
    # -----------------------------------------------------------------

    def _embed_Vm_in_torso(self, Vm_heart: Tensor) -> Tensor:
        """
        Map Vm from heart node numbering to torso node numbering.

        Vm_heart : (n_heart,) — single time snapshot
        returns  : (n_torso,) — zero outside heart
        """
        Vm_torso = torch.zeros(self.n_torso, device=self.device, dtype=self.dtype)
        Vm_torso[self.heart_to_torso] = Vm_heart
        return Vm_torso

    def ecg_forward_solve(
        self,
        Vm: Tensor,
        a_tol: float = 1e-10,
        r_tol: float = 1e-10,
        max_iter: int = 20000,
    ) -> Dict[str, Tensor]:
        """
        Compute 12-lead ECG via full forward solve (one linear solve
        per time step).

        For each time step t:
            K_torso  phi_e(t)  =  -K_i_torso  Vm(t)

        Then V_lead(t) = phi_e at electrode combinations.

        Vm : (n_time, n_heart_nodes)
        returns : dict of lead_name -> (n_time,) tensors

        This is slow (one solve per time step) but serves as a
        reference to validate the lead field method.
        """
        Vm = Vm.to(self.device, self.dtype)
        n_time = Vm.shape[0]

        # Pre-allocate electrode potentials
        electrode_names = [k for k in self.electrodes if k != self.ground]
        phi_at = {name: torch.zeros(n_time, device=self.device, dtype=self.dtype)
                  for name in list(self.electrodes.keys())}

        ground_idx = self.electrodes[self.ground]

        for t in range(n_time):
            if t % 50 == 0:
                print(f"  Forward solve: step {t}/{n_time}", flush=True)

            # Embed Vm into torso numbering
            Vm_torso = self._embed_Vm_in_torso(Vm[t])

            # RHS = -K_i_torso @ Vm_torso
            rhs = -(self.K_i_torso @ Vm_torso)

            # Solve
            phi_e = solve_singular_system(
                self.K_torso, rhs,
                ground_node=ground_idx,
                device=self.device, dtype=self.dtype,
                a_tol=a_tol, r_tol=r_tol, max_iter=max_iter,
            )

            # Read off electrode potentials
            for name, node_id in self.electrodes.items():
                phi_at[name][t] = phi_e[node_id]

        return self._derive_12lead_from_unipolar(phi_at)

    # -----------------------------------------------------------------
    # 12-Lead derivation (common to both methods)
    # -----------------------------------------------------------------

    def _derive_12lead_from_unipolar(
        self, phi: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Derive 12-lead ECG from unipolar potentials at the 10 electrodes.
        """
        ra = phi["RA"]
        la = phi["LA"]
        ll = phi["LL"]

        wct = (ra + la + ll) / 3.0

        return {
            "I":    la - ra,
            "II":   ll - ra,
            "III":  ll - la,
            "aVR":  ra - 0.5 * (la + ll),
            "aVL":  la - 0.5 * (ra + ll),
            "aVF":  ll - 0.5 * (ra + la),
            "V1":   phi["V1"] - wct,
            "V2":   phi["V2"] - wct,
            "V3":   phi["V3"] - wct,
            "V4":   phi["V4"] - wct,
            "V5":   phi["V5"] - wct,
            "V6":   phi["V6"] - wct,
        }

    def compute_12lead(self, Vm: Tensor) -> Dict[str, Tensor]:
        """
        Compute 12-lead ECG using the lead field method.

        Vm : (n_time, n_heart_nodes)
        """
        Vm = Vm.to(self.device, self.dtype)

        required = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
        missing = [k for k in required if k not in self.electrodes]
        if missing:
            raise RuntimeError(f"Missing electrodes: {missing}")
        missing_lf = [k for k in required if k != self.ground and k not in self.q_vectors]
        if missing_lf:
            raise RuntimeError(f"Lead fields not computed for: {missing_lf}")

        # Compute unipolar potentials via lead field
        phi = {}
        for name in self.electrodes:
            if name == self.ground:
                # Ground electrode: V_RL(t) = Vm^T K_i Z_RL.
                # But Z_RL is identically zero by construction (current in = current out
                # at the same node), so the unipolar potential at the ground is zero.
                phi[name] = torch.zeros(Vm.shape[0], device=self.device, dtype=self.dtype)
            else:
                phi[name] = self.ecg_lead_field(Vm, name)

        return self._derive_12lead_from_unipolar(phi)

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------

    @staticmethod
    def plot_ecg(
        ecg_dict: Dict[str, Tensor],
        dt_ms: float = 1.0,
        filename: str = "ecg_12lead.png",
    ):
        """Plot 12-lead ECG in standard 3x4 clinical layout."""
        order = [
            "I",   "aVR", "V1", "V4",
            "II",  "aVL", "V2", "V5",
            "III", "aVF", "V3", "V6",
        ]
        fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=True)

        for ax, lead in zip(axes.flatten(), order):
            sig = ecg_dict[lead].detach().cpu().numpy()
            t = np.arange(len(sig)) * dt_ms
            ax.plot(t, sig, "k-", lw=0.8)
            ax.set_title(lead, fontsize=10, fontweight="bold")
            ax.set_ylabel("mV", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for ax in axes[-1]:
            ax.set_xlabel("ms", fontsize=8)

        fig.suptitle("12-Lead ECG", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {filename}")


# =====================================================================
#  Example usage
# =====================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Step 0: Build the heart-to-torso node mapping
    # ------------------------------------------------------------------
    # This assumes you have already run TorsoHeartMesh to extract the
    # heart mesh. We reconstruct the mapping from old_to_new.
    
    from torso_heart import TorsoHeartMesh
    thm = TorsoHeartMesh()
    thm.load_torsor_mesh("/data/Bei/Torso/HC2/mesh")
    thm.extract_heart_mesh(
        heart_mesh_dir="/data/Bei/Torso/HC2/heart",
        tags=[24, 25, 34, 35, 36],
    )
    heart_to_torso_node = np.where(thm.old_to_new >= 0)[0]
    np.save("heart_to_torso_node.npy", heart_to_torso_node)
    # ------------------------------------------------------------------

    heart_to_torso_node = np.load("heart_to_torso_node.npy")

    solver = ECGSolver(
        torso_mesh_dir="/data/Bei/Torso/HC2/mesh",
        heart_mesh_dir="/data/Bei/Torso/HC2/heart",
        heart_to_torso_node=heart_to_torso_node,
        device=torch.device("cuda:0"),
        dtype=torch.float32,  # float64 recommended for CG accuracy
    )

    # ------------------------------------------------------------------
    # Step 1: Assign conductivities
    # ------------------------------------------------------------------

    # Torso (isotropic, S/m)
    solver.set_torso_conductivity([20, 21], g=0.25)
    solver.set_torso_conductivity(
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23], g=0.6667
    )
    solver.set_torso_conductivity([3], g=0.05)
    solver.set_torso_conductivity([1, 7], g=0.2472)
    solver.set_torso_conductivity([4], g=0.1667)
    solver.set_torso_conductivity([5], g=0.1667)
    solver.set_torso_conductivity([6], g=0.0714)
    solver.set_torso_conductivity([2], g=0.117)
    solver.set_torso_conductivity([8], g=0.1)
    solver.set_torso_conductivity([9], g=0.1)

    # Heart (anisotropic, S/m)
    solver.set_heart_conductivity(
        [24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227
    )
    solver.set_heart_conductivity(
        [34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332
    )

    # ------------------------------------------------------------------
    # Step 2: Assemble stiffness matrices
    # ------------------------------------------------------------------

    solver.build()

    # ------------------------------------------------------------------
    # Step 3: Load electrodes
    # ------------------------------------------------------------------

    solver.load_electrodes("/data/Bei/Torso/HC3/electrodes/lf_src.vtx")

    # ------------------------------------------------------------------
    # Step 4: Load transmembrane potential
    # ------------------------------------------------------------------

    Vm = torch.load("./biventricle/Vm.pt")

    # ------------------------------------------------------------------
    # Step 5A: Lead field method (fast, recommended)
    # ------------------------------------------------------------------

    print("\n=== Lead Field Method ===")
    solver.precompute_lead_fields(I=1.0, a_tol=1e-10, r_tol=1e-10, max_iter=20000)
    ecg_lf = solver.compute_12lead(Vm)
    ECGSolver.plot_ecg(ecg_lf, filename="ecg_lead_field.png")

    # ------------------------------------------------------------------
    # Step 5B: Full forward solve (slow, for validation)
    # ------------------------------------------------------------------
    # Uncomment to cross-validate. This solves one linear system per
    # time step, so it is ~n_time slower than the lead field method.
    #
    # print("\n=== Full Forward Solve (validation) ===")
    # ecg_fwd = solver.ecg_forward_solve(
    #     Vm, a_tol=1e-10, r_tol=1e-10, max_iter=20000
    # )
    # ECGSolver.plot_ecg(ecg_fwd, filename="ecg_forward.png")
    #
    # # Compare
    # for lead in ecg_lf:
    #     diff = (ecg_lf[lead] - ecg_fwd[lead]).abs().max().item()
    #     print(f"  {lead:>4s}: max |LF - FWD| = {diff:.6e}")