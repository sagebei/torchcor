import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torchcor as tc
if torch.cuda.is_available():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
import time
from torchcor.core import *
from pathlib import Path
import pandas as pd
from torchcor.signalanalysis.signalanalysis.ecg_QS import Ecg
from torchcor.tools.igbwriter import IGBWriter
from torchcor.tools.igbreader import IGBReader
from torchcor.core.mesh import region_node_idx


class Monodomain:
    def __init__(self, ionic_models, T, dt, device=None, dtype=None, mass_lumping=False):
        self.device = tc.get_device() if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        # Lump the FEM mass matrix (row-sum onto the diagonal) when True.
        self.mass_lumping = mass_lumping
        
        self.T = T  # ms
        self.dt = dt  # ms
        self.nt = int(T / dt)

        self.ionic_models = ionic_models
        for im in self.ionic_models:
            im.device = self.device
            im.dtype = self.dtype

        self.pcd = None
        self.cg = None

        self.n_nodes = None
        self.nodes = None
        self.elems = None
        self.fibres = None

        self.unique_regions = None

        self.stimuli = None
        self.conductivity = None

        self.K = None
        self.M = None
        self.A = None
        self.Chi = 140 
        self.Cm = 0.01  
        self.theta = 0.5

        self.sigma_i = None
        self.sigma_e = None
        self.simga_m = None

        if torch.cuda.is_available():
            nvmlInit()
            device_id = torch.cuda.current_device()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(device_id)

        self.mesh_path = None
        self.result_path = None


    def load_mesh(self, path="Data/ventricle/Case_1", unit_conversion=1000):
        self.mesh_path = Path(path)
        
        reader = MeshReader(path)
        nodes, elems, regions, fibres = reader.read(unit_conversion=unit_conversion)
        
        self.n_nodes = nodes.shape[0]
        self.nodes = torch.from_numpy(nodes).to(dtype=self.dtype, device=self.device)
        self.elems = elems.to_torch(self.device)
        
        self.regions = torch.from_numpy(regions).to(dtype=torch.long, device=self.device)
        self.unique_regions = torch.unique(self.regions).tolist()
        self.fibres = torch.from_numpy(fibres).to(dtype=self.dtype, device=self.device)

        self.stimuli = Stimuli(self.n_nodes, self.device, self.dtype)
        self.conductivity = Conductivity(self.regions, dtype=self.dtype)
        
        uncovered_regions = self.unique_regions.copy()
        for im in self.ionic_models:
            if len(uncovered_regions) == 0:
                self.ionic_models

            if im.region_ids is None:
                im.region_ids = uncovered_regions.copy()
            uncovered_regions = [id for id in uncovered_regions if id not in set(im.region_ids)]
            
            im.node_indices = region_node_idx(self.elems, im.region_ids)
    
    def add_stimulus(self, vtx_filepath, start, duration, intensity, period=None, count=1):
        if period is None:
            period = self.T
        self.stimuli.add(vtx_filepath, start, duration, intensity, period, count)

    def add_conductivity(self, region_ids, il, it, el=None, et=None):
        if region_ids is None:
            region_ids = self.unique_regions.copy()
        self.conductivity.add(region_ids, il, it, el, et)

    def assemble(self):
        self.sigma_i, self.sigma_e, self.sigma_m = self.conductivity.calculate_sigma(self.fibres)

        if (self.elems.Ln.data is not None) and (self.elems.Tr.data is not None):
            matrices = Matrices1D_3DSurface(vertices=self.nodes, elems=self.elems, device=self.device, dtype=self.dtype)
        elif self.elems.Tr.data is not None:
            matrices = Matrices3DSurface(vertices=self.nodes, triangles=self.elems.Tr.data, device=self.device, dtype=self.dtype)
        elif self.elems.Tt.data is not None:
            matrices = Matrices3D(vertices=self.nodes, tetrahedrons=self.elems.Tt.data, device=self.device, dtype=self.dtype)

        K, M = matrices.assemble_matrices(self.sigma_m)
        
        K = K / self.Chi
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)

        if self.mass_lumping:
            # Replace the consistent mass matrix with its lumped (row-sum)
            Mc = self.M.coalesce()
            rowsum = torch.sparse.sum(Mc, dim=1).to_dense()
            idx = torch.arange(self.n_nodes, device=self.device)
            self.M = torch.sparse_coo_tensor(
                torch.stack([idx, idx]), rowsum,
                (self.n_nodes, self.n_nodes),
                device=self.device, dtype=self.dtype,
            ).coalesce()

        A = self.M * self.Cm + self.K * self.dt * self.theta
        A = A.coalesce()

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)

        self.M = self.M.to_sparse_csr()
        self.K = self.K.to_sparse_csr()
        self.A = A.to_sparse_csr()

        self.cg = ConjugateGradient(self.pcd, self.A, dtype=torch.float64)
        

    def step(self, u, t, a_tol, r_tol, max_iter, verbose=False):
        ionic_time = 0
        electric_time = 0
        if verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

        ### ionic ###
        b = u.clone()
        for im in self.ionic_models:
            idx = im.node_indices
            du = im.differentiate(u[idx]) / 100
            b[idx] = u[idx]* self.Cm + self.dt * du
        #############

        if verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ionic_time = time.time() - start_time
            start_time = time.time()

        ### electric ###
        Istim = self.stimuli.apply(t) / 100
        b += self.dt * Istim

        b = self.M @ b
        b -= (1 - self.theta) * self.dt * self.K @ u

        u, n_iter = self.cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        ################
        if verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            electric_time = time.time() - start_time

        return u, n_iter, ionic_time, electric_time

    
    def solve(self, a_tol, r_tol, max_iter, linear_guess=True, snapshot_interval=5, verbose=True, result_path=None):
        self.result_path = Path(result_path)
        self.snapshot_interval = snapshot_interval

        self.assemble()
        
        u = torch.zeros((self.n_nodes), dtype=self.dtype, device=self.device)
        for im in self.ionic_models:
            u_im = im.initialize(im.node_indices.shape[0])
            u[im.node_indices] = u_im.clone()

        u_initial = u.clone()
        self.cg.initialize(x=u, linear_guess=linear_guess)
        ts_per_frame = int(snapshot_interval / self.dt)

        t = 0
        solving_time = time.time()
        total_ionic_time = 0
        total_electric_time = 0
        n_total_iter = 0
        gpu_utilisation_list = []
        gpu_memory_list = []
        solution_list = [u_initial]
        for n in range(1, self.nt + 1):
            t += self.dt
            
            ### CG step ###
            u, n_iter, ionic_time, electric_time = self.step(u, t, a_tol, r_tol, max_iter, verbose)
            if n_iter >= max_iter:
                raise Exception("exceeded max_iter")

            n_total_iter += n_iter
            total_ionic_time += ionic_time
            total_electric_time += electric_time
    
            ### keep track of GPU usage ###
            if n % ts_per_frame == 0:
                solution_list.append(u.clone())

                if torch.cuda.is_available():
                    gpu_utilisation_list.append(nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu)
                    gpu_memory_list.append(nvmlDeviceGetMemoryInfo(self.gpu_handle).used / 1e9)
                
                if verbose and snapshot_interval != self.T:
                    print(f"t: {round(t, 1)}/{self.T} |", 
                          f"Time elapsed: {round(time.time() - solving_time, 1)} |", 
                          f"CG iter:", n_total_iter,
                          flush=True)
                n_total_iter = 0

        ### print log info to console ###
        if verbose:
            if torch.cuda.is_available():
                total_time = time.time() - solving_time
                gpu_util = sum(gpu_utilisation_list) / len(gpu_utilisation_list)
                gpu_mem = sum(gpu_memory_list) / len(gpu_memory_list)

                print(
                    f"nodes: {self.n_nodes} | "
                    f"total_time: {total_time:.2f}s | "
                    f"ionic_time: {total_ionic_time:.2f}s | "
                    f"electric_time: {total_electric_time:.2f}s | "
                    f"gpu_util: {gpu_util:.2f}% | "
                    f"gpu_mem: {gpu_mem:.2f}GB",
                    flush=True)
            else:
                total_time = time.time() - solving_time
                print(
                    f"nodes: {self.n_nodes} | "
                    f"total_time: {total_time:.2f}s | "
                    f"ionic_time: {total_ionic_time:.2f}s | "
                    f"electric_time: {total_electric_time:.2f}s",
                    flush=True)

        Vm = torch.stack(solution_list, dim=0)
        return Vm

    def compute_activation_map_dvdt(
        self,
        Vm: torch.Tensor,
        snapshot_interval: float = 1.0,
        threshold: float = 30.0,
        interpolate: bool = False,
        save: bool = False,
    ) -> torch.Tensor:
        T, N = Vm.shape

        dVdt = (Vm[1:] - Vm[:-1]) / snapshot_interval
        peak_idx = torch.argmax(dVdt, dim=0)
        max_dVdt = dVdt.gather(0, peak_idx.unsqueeze(0)).squeeze(0)

        valid = max_dVdt >= threshold

        if interpolate:
            p = peak_idx.clamp(1, dVdt.shape[0] - 2)
            y0 = dVdt.gather(0, (p - 1).unsqueeze(0)).squeeze(0)
            y1 = dVdt.gather(0, p.unsqueeze(0)).squeeze(0)
            y2 = dVdt.gather(0, (p + 1).unsqueeze(0)).squeeze(0)
            curvature = y0 - 2 * y1 + y2
            shift = torch.where(
                curvature < -1e-12,
                0.5 * (y0 - y2) / curvature.clamp(max=-1e-12),
                torch.zeros_like(y1),
            ).clamp(-1.0, 1.0)
            at_idx = p.float() + shift
            edge = (peak_idx == 0) | (peak_idx >= dVdt.shape[0] - 1)
            at_idx = torch.where(edge, peak_idx.float(), at_idx)
            ATs = at_idx * snapshot_interval
        else:
            ATs = peak_idx.float() * snapshot_interval

        ATs[~valid] = float('nan')

        if save:
            self.result_path.mkdir(parents=True, exist_ok=True)
            torch.save(ATs.cpu(), self.result_path / "ATs.pt")

        return ATs


    def compute_activation_map(
        self,
        Vm: torch.Tensor,
        snapshot_interval: float = 1.0,
        threshold: float = 0,
        interpolate: bool = False,
        save: bool = False
    ) -> torch.Tensor:
        T, N = Vm.shape

        above = Vm > threshold
        ascending = ~above[:-1] & above[1:]
        has_crossing = ascending.any(dim=0)

        crossing = torch.argmax(ascending.to(torch.uint8), dim=0)

        if interpolate:
            t0 = crossing.unsqueeze(0)
            v0 = Vm.gather(0, t0).squeeze(0)
            v1 = Vm.gather(0, (t0 + 1).clamp_max(T - 1)).squeeze(0)
            denom = (v1 - v0).clamp(min=1e-12)
            frac = ((threshold - v0) / denom).clamp(0.0, 1.0)
            ATs = (crossing.float() + frac) * snapshot_interval
        else:
            ATs = crossing.float() * snapshot_interval

        ATs[~has_crossing] = float('nan')

        if save:
            self.result_path.mkdir(parents=True, exist_ok=True)
            torch.save(ATs.cpu(), self.result_path / "ATs.pt")
        return ATs


    def compute_repolarization_map(
        self,
        Vm: torch.Tensor,
        snapshot_interval: float = 1.0,
        threshold: float = -70.0,
        search_after: torch.Tensor = None,
        first: bool = True,
        interpolate: bool = False,
        save: bool = False
    ) -> torch.Tensor:
        T, N = Vm.shape
        device = Vm.device

        above = Vm > threshold
        descending = above[:-1] & ~above[1:]

        if search_after is not None:
            search_after = search_after.to(device)
            never_activated = torch.isnan(search_after)
            safe = torch.where(never_activated, torch.zeros_like(search_after), search_after)
            act_idx = (safe / snapshot_interval).long().clamp_(0, T - 2)
            act_idx[never_activated] = T - 1
            time_idx = torch.arange(T - 1, device=device).unsqueeze(1)
            descending = descending & (time_idx >= act_idx.unsqueeze(0))
            descending[:, never_activated] = False

        has_crossing = descending.any(dim=0)

        if first:
            crossing = torch.argmax(descending.to(torch.uint8), dim=0)
        else:
            flipped = descending.flip(0).to(torch.uint8)
            crossing = (T - 2) - torch.argmax(flipped, dim=0)

        if interpolate:
            t0 = crossing.unsqueeze(0)
            v0 = Vm.gather(0, t0).squeeze(0)
            v1 = Vm.gather(0, (t0 + 1).clamp_max(T - 1)).squeeze(0)
            denom = (v0 - v1).clamp(min=1e-12)
            frac = ((v0 - threshold) / denom).clamp(0.0, 1.0)
            RTs = (crossing.float() + frac) * snapshot_interval
        else:
            RTs = (crossing.float() + 1.0) * snapshot_interval

        RTs[~has_crossing] = float('nan')

        if save:
            self.result_path.mkdir(parents=True, exist_ok=True)
            torch.save(RTs.cpu(), self.result_path / "RTs.pt")

        return RTs


    def save_vm(self, Vm):
        self.result_path.mkdir(parents=True, exist_ok=True)
        torch.save(Vm.cpu(), self.result_path / "Vm.pt")

    def vm_to_vtk(self, Vm=None, step=1):
        self.result_path.mkdir(parents=True, exist_ok=True)

        if self.elems.Tr.data is not None:
            visualization = VTK3DSurface(self.nodes, self.elems.Tr.data)
        elif self.elems.Tt.data is not None:
            visualization = VTK3D(self.nodes, self.elems.Tt.data)
        
        if Vm is not None:
            Vm = Vm.cpu().numpy().astype(np.float32)
            n_solutions = Vm.shape[0]
            for i in range(0, n_solutions, step):
                visualization.save_frame(color_values=Vm[i],
                                         frame_path=self.result_path / f"Vm_vtk/ms_{i * self.snapshot_interval}.vtk")

        ATS_path = self.result_path / "ATs.pt"
        if ATS_path.exists():
            ATs = torch.load(ATS_path)
            visualization.save_frame(color_values=ATs,
                                     frame_path=self.result_path / "ATs.vtk")
        
        RTs_path = self.result_path / "RTs.pt"
        if RTs_path.exists():
            RTs = torch.load(RTs_path)
            visualization.save_frame(color_values=RTs,
                                     frame_path=self.result_path / "RTs.vtk")


    def vm_to_igb(self, Vm, filename: str = "Vm.igb"):
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        data = Vm.cpu().numpy().astype(np.float32)  #ntXnx
        
        header = {'x':self.n_nodes, 'y':1, 'z':1,
                  't': data.shape[0], 
                  'org_t': 0.0, 
                  'Tend': self.T}
        
        im = IGBWriter()
        im.initialise_from_data(header, data)
        im.set_fname(os.path.join(self.result_path, filename))
        im.write_data_to_file()


    def igb_to_vtk(self, igb_path, step=1):
        reader = IGBReader()
        reader.read(igb_path)
        Vms = torch.from_numpy(reader.data())

        if (self.elems.Ln is None) and (self.elems.Tr is not None):
            visualization = VTK3DSurface(self.nodes, self.elems.Tr)
        elif self.elems.Tt is not None:
            visualization = VTK3D(self.nodes, self.elems.Tt)
        
        n_solutions = Vms.shape[0]
        for i in range(0, n_solutions, step):
            visualization.save_frame(color_values=Vms[i],
                                     frame_path=self.result_path / f"igb_vtk/frame_{i}.vtk")
        

    def phie_recovery(self, Vm=None, a_tol=1e-5, r_tol=1e-5, max_iter=1000, verbose=False):
        """Recover the extracellular potential phi_e from the monodomain Vm.

        Solves the pseudo-bidomain elliptic problem  K_ie phi_e = -K_i Vm  per
        time frame, where K_ie is assembled with (sigma_i + sigma_e) and K_i with
        sigma_i, gauged to zero mean (the operator is singular up to a constant).
        This is the same formulation openCARP uses for its pseudo-bidomain phi_e.

        Requires bidomain conductivities (il, it, el, et).  Pass ``Vm`` directly
        or leave it ``None`` to load ``result_path/Vm.pt``.  Returns the
        ``(T, N_nodes)`` phi_e field (also saved to ``result_path/Phi_e.pt``).
        """
        if self.sigma_i is None or self.sigma_e is None:
            self.sigma_i, self.sigma_e, self.sigma_m = self.conductivity.calculate_sigma(self.fibres)
        if self.sigma_i is None or self.sigma_e is None:
            raise Exception("phie_recovery needs bidomain conductivities (il, it, el, et).")

        if Vm is None:
            Vm = torch.load(self.result_path / "Vm.pt")
        Vm = Vm.to(self.device)

        def stiffness(sigma):
            if (self.elems.Ln.data is not None) and (self.elems.Tr.data is not None):
                matrices = Matrices1D_3DSurface(vertices=self.nodes, elems=self.elems, device=self.device, dtype=self.dtype)
            elif self.elems.Tr.data is not None:
                matrices = Matrices3DSurface(vertices=self.nodes, triangles=self.elems.Tr.data, device=self.device, dtype=self.dtype)
            else:
                matrices = Matrices3D(vertices=self.nodes, tetrahedrons=self.elems.Tt.data, device=self.device, dtype=self.dtype)
            K, _ = matrices.assemble_matrices(sigma)
            return K.coalesce()

        K_ie = stiffness(self.sigma_i + self.sigma_e)   # (sigma_i + sigma_e) stiffness
        K_i = stiffness(self.sigma_i)                   # sigma_i stiffness

        pcd = Preconditioner()
        pcd.create_Jocobi(K_ie)
        cg = ConjugateGradient(pcd, K_ie, dtype=torch.float64)
        cg.initialize(x=torch.zeros_like(Vm[0]), linear_guess=True)

        K_ie = K_ie.to_sparse_csr()
        K_i = K_i.to_sparse_csr()

        phie_list = []
        for i in range(Vm.shape[0]):
            b = -K_i @ Vm[i, :]
            phi_e, n_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            phi_e = phi_e.clone()                       # detach from CG's reused internal buffer
            phi_e -= phi_e.mean()                       # gauge the constant null-space
            phie_list.append(phi_e)
            if verbose:
                print(f"phi_e frame {i}: [{phi_e.min().item():.3f}, {phi_e.max().item():.3f}] | CG {n_iter}", flush=True)

        phi_e_all = torch.stack(phie_list, dim=0)
        if self.result_path is not None:
            self.result_path.mkdir(parents=True, exist_ok=True)
            torch.save(phi_e_all.cpu(), self.result_path / "Phi_e.pt")
        return phi_e_all


    def simulated_ECG(self):
        im = IGBWriter({
            "fname": os.path.join(self.result_path, "phie.igb"),
            "Tend": self.T + 1,
            "nt": self.nt + 1,
            "nx": self.n_nodes,
            "ny": 1,
            "nz": 1
        })

        path = os.path.join(self.result_path, "Phi_e.pt")
        phie = torch.load(path).numpy()
        for p in phie:
           im.imshow(p)
        
        ECGs = Ecg(os.path.join(self.result_path, 'phie.igb'), dt=1)

        lp, hp = 100, 0.01
        ECGs.filter = 'butterworth'
        ECGs.apply_filter(freq_filter=lp, order=2, sample_freq=1000, filter_type='low')
        ECGs.apply_filter(freq_filter=hp, order=2, sample_freq=1000, filter_type='high')
        
        ECGspd = pd.DataFrame(ECGs.data)
        print(ECGspd.columns)
        ECGspd.to_csv(self.result_path / 'simulated_filtered.dat', sep=' ', header=False, mode='w')



