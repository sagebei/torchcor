import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torchcor as tc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
import time
from torchcor.core import *
from pathlib import Path
import pandas as pd
from torchcor.signalanalysis.signalanalysis.ecg_QS import Ecg
from torchcor.tools.igbwriter import IGBWriter


class Monodomain:
    def __init__(self, ionic_model, T, dt, device=tc.get_device(), dtype=None):
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        
        self.T = T  # ms
        self.dt = dt  # ms
        self.nt = int(T / dt)

        self.ionic_model = ionic_model

        self.pcd = None
        self.cg = None

        self.n_nodes = None
        self.nodes = None
        self.elems = None
        self.fibres = None
        self.region_ids = None

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
        self.elems = torch.from_numpy(elems).to(dtype=torch.int, device=self.device)
        self.regions = torch.from_numpy(regions).to(dtype=torch.int, device=self.device)
        self.fibres = torch.from_numpy(fibres).to(dtype=self.dtype, device=self.device)

        self.stimuli = Stimuli(self.n_nodes, self.device, self.dtype)
        self.conductivity = Conductivity(self.regions, dtype=self.dtype)

    def add_stimulus(self, vtx_filepath, start, duration, intensity, period=None, count=1):
        if period is None:
            period = self.T
        self.stimuli.add(vtx_filepath, start, duration, intensity, period, count)

    def add_conductivity(self, region_ids, il, it, el=None, et=None):
        self.conductivity.add(region_ids, il, it, el, et)

    def assemble(self):
        self.sigma_i, self.sigma_e, self.sigma_m = self.conductivity.calculate_sigma(self.fibres)

        if self.elems.shape[1] == 3:
            matrices = Matrices3DSurface(vertices=self.nodes, triangles=self.elems, device=self.device, dtype=self.dtype)
        else:
            matrices = Matrices3D(vertices=self.nodes, tetrahedrons=self.elems, device=self.device, dtype=self.dtype)

        K, M = matrices.assemble_matrices(self.sigma_m)
        
        K = K / self.Chi
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
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
            torch.cuda.synchronize()
            start_time = time.time()

        ### ionic ###
        du = self.ionic_model.differentiate(u) / 100
        b = u * self.Cm + self.dt * du
        #############

        if verbose:
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
            torch.cuda.synchronize()
            electric_time = time.time() - start_time

        return u, n_iter, ionic_time, electric_time

    def solve(self, a_tol, r_tol, max_iter, calculate_AT_RT=True, linear_guess=True, snapshot_interval=1, verbose=True, result_path=None):
        self.result_path = Path(result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.assemble()
        
        u = self.ionic_model.initialize(self.n_nodes)
        u_initial = u.clone()
        self.cg.initialize(x=u, linear_guess=linear_guess)
        ts_per_frame = int(snapshot_interval / self.dt)
    
        if calculate_AT_RT:
            activation_time = torch.ones_like(u_initial) * -1
            repolarization_time = torch.ones_like(u_initial) * -1

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
            n_total_iter += n_iter
            total_ionic_time += ionic_time
            total_electric_time += electric_time
            
            ### calculate AT and RT ###
            if calculate_AT_RT:
                activation_time[(u > 0) & (activation_time == -1)] = t
                repolarization_time[(activation_time > 0) & (repolarization_time == -1) & (u < -70)] = t

                # mask_peak_update = (u > u_peak) & (activation_time > 0)
                # u_peak[mask_peak_update] = u[mask_peak_update]
                # repolarization_threshold = u_initial + 0.1 * (u_peak - u_initial)
                # repolarization_time[(u < repolarization_threshold) &
                #                     (repolarization_time == 0) &
                #                     (activation_time > 0)] = t
            
            ### keep track of GPU usage ###
            if n % ts_per_frame == 0:
                solution_list.append(u.clone())

                gpu_utilisation_list.append(nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu)
                gpu_memory_list.append(nvmlDeviceGetMemoryInfo(self.gpu_handle).used / 1e9)
                
                if verbose and snapshot_interval != self.T:
                    print(f"t: {round(t, 1)}/{self.T}", 
                          f"Time elapsed:", round(time.time() - solving_time, 2),
                          f"Total CG iter:", n_total_iter,
                          flush=True)

        ### save AT, RT, and solutions to disk ###
        if calculate_AT_RT:
            torch.save(activation_time.cpu(), self.result_path / "ATs.pt")
            torch.save(repolarization_time.cpu(), self.result_path / "RTs.pt")

            # torch.save(self.K.to_sparse_coo().cpu(), self.result_path / "K.pt")
            # torch.save(self.M.to_sparse_coo().cpu(), self.result_path / "M.pt")
            # torch.save(self.A.to_sparse_coo().cpu(), self.result_path / "A.pt")

        if snapshot_interval < self.T:
            torch.save(torch.stack(solution_list, dim=0).cpu(), self.result_path / "Vm.pt")

        ### print log info to console ###
        if verbose:
            print(self.ionic_model.name,
                  self.n_nodes, 
                  round(time.time() - solving_time, 2),
                  round(total_ionic_time, 2),
                  round(total_electric_time, 2),
                  n_total_iter,
                  f"{round(sum(gpu_utilisation_list)/len(gpu_utilisation_list), 2)}",
                  f"{round(sum(gpu_memory_list)/len(gpu_memory_list), 2)}",
                  flush=True)
            
            if calculate_AT_RT:
                print("ATs: ", activation_time.cpu().min().item(), activation_time.cpu().max().item(), flush=True)
                print("RTs: ", repolarization_time.cpu().min().item(), repolarization_time.cpu().max().item(), flush=True)

        return n_total_iter


    def pt_to_vtk(self):
        start_time = time.time()

        if self.elems.shape[1] == 3:
            visualization = VTK3DSurface(self.nodes, self.elems)
        else:
            visualization = VTK3D(self.nodes, self.elems)
        
        solutions = torch.load(self.result_path / "Vm.pt", map_location=torch.device('cpu'))
        n_solutions = solutions.shape[0]
        for i in range(n_solutions):
            visualization.save_frame(color_values=solutions[i],
                                     frame_path=self.result_path / f"Vm_vtk/frame_{i}.vtk")
        

        ATs = torch.load(self.result_path / "ATs.pt")
        visualization.save_frame(color_values=ATs,
                                 frame_path=self.result_path / "ATs.vtk")
        RTs = torch.load(self.result_path / "RTs.pt")
        visualization.save_frame(color_values=RTs,
                                 frame_path=self.result_path / "RTs.vtk")

        print(f"Saved vtk files in {round(time.time() - start_time, 2)}", flush=True)


    def phie_recovery(self, a_tol=1e-5, r_tol=1e-5, max_iter=100):
        Vm = torch.load(self.result_path / "Vm.pt").to(self.device)

        if self.elems.shape[1] == 3:
            matrices = Matrices3DSurface(vertices=self.nodes, triangles=self.elems, device=self.device, dtype=self.dtype)
        else:
            matrices = Matrices3D(vertices=self.nodes, tetrahedrons=self.elems, device=self.device, dtype=self.dtype)
        K_ie, _ = matrices.assemble_matrices(self.sigma_i + self.sigma_e)
        K_i, _ = matrices.assemble_matrices(self.sigma_i)

        pcd = Preconditioner()
        pcd.create_Jocobi(K_ie)
        cg = ConjugateGradient(pcd, K_ie, dtype=torch.float64)
        cg.initialize(x=torch.zeros_like(Vm[0]), linear_guess=True)

        K_ie = K_ie.to_sparse_csr()
        K_i = K_i.to_sparse_csr()
        
        phie_list = []
        for i in range(Vm.shape[0]):
            V = Vm[i, :]
            
            b = -K_i @ V
            phi_e, n_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            
            phi_e -= phi_e.mean()
            phie_list.append(phi_e)

            print(phi_e.min().item(), phi_e.max().item(), n_iter)
        
        phi_e_all = torch.stack(phie_list, dim=0)
        torch.save(phi_e_all.cpu(), self.result_path / "Phi_e.pt")
        print(phi_e_all.min().item(), phi_e_all.max().item())


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
        

    def pt_to_igb(self, filename: str = "vm.igb"):
        ''' This function converts the output to an igb file format
        '''
        data = torch.load(self.result_path / "Vm.pt", map_location=torch.device('cpu')).numpy().astype(np.float32)  #ntXnx
        
        header = {'x':self.n_nodes, 'y':1, 'z':1,
                  't': data.shape[0], 
                  'org_t': 0.0, 
                  'Tend': self.T}
        
        im = IGBWriter()
        im.initialise_from_data(header, data)
        im.set_fname(os.path.join(self.result_path, filename))
        im.write_data_to_file()
        
        



