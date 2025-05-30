import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from core.assemble import Matrices3DSurface
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.visualize import VTK3DSurface
from core.reorder import RCM as RCM
import time
from tools.triangulation import Triangulation
from tools.materialproperties import MaterialProperties
from tools.stimulus import Stimulus
from utils import load_stimulus_region


class AtriumSimulatorMitchell:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=None):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float64

        self.T = T  # ms = 2.4s
        self.dt = dt  # ms
        self.nt = int(T // dt)
        self.rcm = RCM(device=device, dtype=dtype) if apply_rcm else None

        self.ionic_model = ionic_model

        self.pcd = None

        self.point_region_ids = None
        self.n_nodes = None
        self.vertices = None
        self.triangles = None
        self.fibers = None

        self.material_config = None

        self.K = None
        self.M = None
        self.A = None

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self, path="/Users/bei/Project/FinitePDE/data/Case_1"):
        mesh = Triangulation()
        mesh.readMesh(path)

        self.point_region_ids = mesh.point_region_ids()
        self.n_nodes = self.point_region_ids.shape[0]

        self.vertices = torch.from_numpy(mesh.Pts()).to(dtype=self.dtype, device=self.device)
        self.triangles = torch.from_numpy(mesh.Elems()['Trias'][:, :-1]).to(dtype=torch.long, device=self.device)
        self.fibers = torch.from_numpy(mesh.Fibres()).to(dtype=self.dtype, device=self.device)

        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.triangles)

    def add_material_property(self, material_config):
        self.material_config = material_config

        material = MaterialProperties()

        material.add_nodal_property('tau_in', 'uniform', material_config["tin"])
        material.add_nodal_property('tau_out', 'uniform', material_config["tout"])
        material.add_nodal_property('tau_open', 'uniform', material_config["topen"])
        material.add_nodal_property('tau_close', 'uniform', material_config["tclose"])
        material.add_nodal_property('u_gate', 'uniform', material_config["vg"])
        material.add_nodal_property('u_crit', 'uniform', material_config["vg"])

        nodal_properties = material.nodal_property_names()

        for npr in nodal_properties:
            npr_type = material.nodal_property_type(npr)
            attribute_value = self.ionic_model.get_attribute(npr)

            if attribute_value is not None:
                if npr_type == "uniform":
                    values = material.NodalProperty(npr, -1, -1)
                else:
                    values = torch.full(size=(self.n_nodes, 1), fill_value=attribute_value)
                    for point_id, region_id in enumerate(self.point_region_ids):
                        values[point_id] = material.NodalProperty(npr, point_id, region_id)
                self.ionic_model.set_attribute(npr, values)

    def set_stimulus_region(self, path="/Users/bei/Project/FinitePDE/data/Case_1.vtx"):
        self.stimulus_region = load_stimulus_region(path)  # (2168,)

    def add_stimulus(self, stim_config):
        S = torch.zeros(size=(self.n_nodes,), device=self.device, dtype=torch.bool)
        S[self.stimulus_region] = True
        if self.rcm is not None:
            S = self.rcm.reorder(S)

        stimulus = Stimulus(stim_config)
        stimulus.set_stimregion(S)

        self.stimuli.append(stimulus)

    def assemble(self):
        if self.rcm is not None:
            rcm_vertices = self.rcm.reorder(self.vertices)
            rcm_triangles = self.rcm.map(self.triangles)
        else:
            rcm_vertices = self.vertices
            rcm_triangles = self.triangles

        # region_ids = domain.Elems()['Trias'][:, -1]
        sigma_l = self.material_config["diffusl"]
        sigma_t = self.material_config["diffust"]
        sigma = sigma_t * torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self.fibers.shape[0], 3, 3)
        sigma += (sigma_l - sigma_t) * self.fibers.unsqueeze(2) @ self.fibers.unsqueeze(1)

        matrices = Matrices3DSurface(vertices=rcm_vertices, triangles=rcm_triangles, device=self.device, dtype=self.dtype)
        K, M = matrices.assemble_matrices(sigma)
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M + self.K * self.dt

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)
        self.A = A.to_sparse_csr()

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u = self.ionic_model.initialize(self.n_nodes, self.dt)
        if self.rcm is not None:
            u = self.rcm.reorder(u)

        cg = ConjugateGradient(self.pcd)
        cg.initialize(x=u)

        ts_per_frame = int(plot_interval / self.dt)
        ctime = 0
        visualization = VTK3DSurface(self.vertices.cpu(), self.triangles.cpu())
        solving_time = time.time()
        for n in range(1, self.nt + 1):
            ctime += self.dt
            du = self.ionic_model.differentiate(u)
            b = u + self.dt * du
            for stimulus in self.stimuli:
                I0 = stimulus.stimApp(ctime)
                b += self.dt * I0
            b = self.M @ b

            u, total_iter = cg.solve(self.A, b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)

            if total_iter == max_iter:
                raise Exception(f"The solution did not converge at {n}th timestep")
            if verbose:
                print(f"{n} / {self.nt + 1}: {total_iter}; {round(time.time() - solving_time, 2)}")

            if n % ts_per_frame == 0:
                visualization.save_frame(color_values=self.rcm.inverse(u).cpu().numpy() if self.rcm is not None else u.cpu().numpy(),
                                         frame_path=f"./vtk_files_{self.n_nodes}_{self.rcm is not None}/frame_{n}.vtk")

        print(f"Solved in {round(time.time() - solving_time, 2)} seconds")




if __name__ == "__main__":
    from simulator import AtriumSimulatorMitchell
    from ionic import MitchellSchaeffer
    import torch
    from pathlib import Path


    simulation_time = 2400
    dt = 0.01
    stim_config = {'tstart': 0.0,
                'nstim': 3,
                'period': 800,
                'duration': max([2.0, dt]),
                'intensity': 1.0,
                'name': 'S1'}
    material_config = {"diffusl": (1000 * 1000) * 0.175,
                    "diffust": (1000 * 1000) * 0.04375}

    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    home_directory = Path.home()

    ionic_model = MitchellSchaeffer(device=device)
    simulator = AtriumSimulatorMitchell(ionic_model, T=simulation_time, dt=dt, apply_rcm=True, device=device)
    simulator.load_mesh(path=f"{home_directory}/Data/atrium/Case_1")
    simulator.add_material_property(material_config)
    simulator.set_stimulus_region(path=f"{home_directory}/Data/atrium/Case_1.vtx")
    simulator.add_stimulus(stim_config)
    simulator.assemble()
    simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)