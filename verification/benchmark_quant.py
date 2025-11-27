import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


import pygmsh
from TenTusscherPanfilov import TenTusscherPanfilov
import torch
from torchcor.core.assemble import Matrices3D
from torchcor.core.preconditioner import Preconditioner
from torchcor.core.solver import ConjugateGradient
from torchcor.core.visualize import VTK3D
from torchcor.core.reorder import RCM as RCM
import numpy as np
import time
import matplotlib.pyplot as plt


class Monodomain:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=None):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float64

        self.T = T  # ms 
        self.dt = dt  # ms
        self.dx = None
        self.nt = int(T // dt)
        self.rcm = RCM(device=device, dtype=dtype) if apply_rcm else None

        self.ionic_model = ionic_model
        self.ionic_model.construct_tables()

        self.pcd = None

        self.point_region_ids = None
        self.n_nodes = None
        self.vertices = None
        self.tetrahedral = None
        self.fibers = None
        self.region_ids = None

        self.material_config = None

        
        self.Chi = 140  # mm
        self.Cm = 0.01  # uFmm
        self.K = None
        self.M = None
        self.A = None
        self.theta = 0.5

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self, dx):
        self.dx = dx

        length = 20  # mm
        width = 7    # mm
        height = 3   # mm

        # Create the cube with tethrahedrals
        with pygmsh.geo.Geometry() as geom:
            geom.add_box(
                x0=0, x1=length,  
                y0=0, y1=width,  
                z0=0, z1=height, 
                mesh_size=dx
            )
            mesh = geom.generate_mesh()
        
        self.vertices = torch.from_numpy(mesh.points).to(dtype=self.dtype, device=self.device)
        self.n_nodes = self.vertices.shape[0]
        self.tetrahedral = torch.from_numpy(mesh.cells_dict["tetra"]).to(dtype=torch.long, device=self.device)
        self.fibers = torch.tensor([1, 0, 0]).repeat(self.tetrahedral.shape[0], 1).to(dtype=self.dtype, device=self.device)

        corner_size = 1.5
        self.corner_indices = torch.where((self.vertices[:, 0] <= corner_size) &  # x <= 1.5
                                       (self.vertices[:, 1] <= corner_size) &  # y <= 1.5
                                       (self.vertices[:, 2] <= corner_size)    # z <= 1.5
                                )[0] 
        
        self.P1_index = torch.where((self.vertices[:, 0] == 0) &  
                                    (self.vertices[:, 1] == 0) &  
                                    (self.vertices[:, 2] == 0)    
                                )[0]
        self.P8_index = torch.where((self.vertices[:, 0] == 20) &  
                                    (self.vertices[:, 1] == 7) &  
                                    (self.vertices[:, 2] == 3)   
                                )[0]
        

        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        t_x = x / 20  
        t_y = y / 7  
        t_z = z / 3  
        if dx == 0.1:
            tolerance = 0.001
        if dx == 0.2:
            tolerance = 0.01
        elif dx == 0.5:
            tolerance = 0.06

        mask = (torch.abs(t_x - t_y) < tolerance) & \
               (torch.abs(t_y - t_z) < tolerance) & \
               (t_x >= 0) & \
               (t_x <= 1)
        diagonal_indices = torch.where(mask)[0]
        diagonal_vertices = self.vertices[diagonal_indices]
        distances = torch.sqrt((diagonal_vertices[:, 0] - 0) ** 2 + (diagonal_vertices[:, 1] - 0) ** 2 + (diagonal_vertices[:, 2] - 0) ** 2)
        sorted_indices = torch.argsort(distances)
        self.diagonal_indices = diagonal_indices[sorted_indices]
        self.diagonal_distance = distances[sorted_indices]

        print(self.vertices.shape, self.tetrahedral.shape, self.fibers.shape)

        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.tetrahedral)

    def add_material_property(self, material_config):
        self.material_config = material_config

    def assemble(self):
        if self.rcm is not None:
            rcm_vertices = self.rcm.reorder(self.vertices)
            rcm_tetrahedral = self.rcm.map(self.tetrahedral)
        else:
            rcm_vertices = self.vertices
            rcm_tetrahedral = self.tetrahedral

        sigma_l = self.material_config['diffusl']
        sigma_t = self.material_config['diffust']

        sigma = sigma_t * torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self.fibers.shape[0], 3, 3)
        sigma += (sigma_l - sigma_t) * self.fibers.unsqueeze(2) @ self.fibers.unsqueeze(1)

        matrices = Matrices3D(vertices=rcm_vertices, tetrahedrons=rcm_tetrahedral, device=self.device, dtype=self.dtype)
        K, M = matrices.assemble_matrices(sigma)

        K = K / self.Chi
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M * self.Cm + self.K * self.dt * self.theta

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)
        self.A = A.to_sparse_csr()
        self.M = self.M.to_sparse_csr()
        self.K = self.K.to_sparse_csr()

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u = self.ionic_model.initialize(self.n_nodes)

        stimulus = torch.zeros_like(u)
        stimulus[self.corner_indices] = 50

        if self.rcm is not None:
            u = self.rcm.reorder(u)

        cg = ConjugateGradient(self.pcd, self.A, dtype=self.dtype)
        cg.initialize(x=u)

        ctime = 0
        n_total_iter = 0
        self.activation_time = torch.zeros_like(self.diagonal_indices, dtype=torch.float32)

        u_plot = []
        for n in range(1, self.nt + 1):
            ctime += self.dt

            du = self.ionic_model.differentiate(u) / 100
            
            b = u * self.Cm + self.dt * du

            # apply the stimulus for 2 mm
            if ctime <= 2.0:
                b += self.dt * stimulus / 100

            b = self.M @ b 
            b -= (1 - self.theta) * self.dt * self.K @ u
            
            u, n_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            n_total_iter += n_iter

            self.activation_time[(u[self.diagonal_indices] > 0) & (self.activation_time == 0)] = ctime

            if n_iter == max_iter:
                raise Exception(f"The solution did not converge at {n}th timestep")
            
            if verbose:
                print(f"{ctime:.3f} / {self.T}: {n_iter}; ", 
                      round(u[self.P1_index].item(), 1), 
                      round(u[self.P8_index].item(), 1))
                
            if u[self.P8_index].item() > 0:
                break

if __name__ == "__main__":
    dt = 0.005  # ms

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    ionic_model = TenTusscherPanfilov(cell_type="EPI", dt=dt, device=device, dtype=dtype)
    ionic_model.V_init = -85.23
    ionic_model.Xr1_init = 0.00621
    ionic_model.Xr2_init = 0.4712
    ionic_model.Xs_init = 0.0095
    ionic_model.M_init = 0.00172
    ionic_model.H_init = 0.7444
    ionic_model.J_init = 0.7045
    ionic_model.d_init = 3.373e-5
    ionic_model.F_init = 0.7888
    ionic_model.F2_init = 0.9755
    ionic_model.FCaSS_init = 0.9953
    ionic_model.S_init = 0.999998
    ionic_model.R_init = 2.42e-8
    ionic_model.Cai_init = 0.000126
    ionic_model.CaSR_init = 3.64
    ionic_model.CaSS_init = 0.00036
    ionic_model.R__init = 0.9073
    ionic_model.Nai_init = 8.064
    ionic_model.Ki_init = 136.89

    il = 0.17 
    it = 0.019 
    el = 0.62 
    et = 0.2 
    material_config = {"diffusl": il * el * (1 / (il + el)),
                       "diffust": it * et * (1 / (it + et))}

    simulator = Monodomain(ionic_model, 
                           T=150, 
                           dt=dt, 
                           apply_rcm=False, 
                           device=device, 
                           dtype=dtype)
    plt.subplots(figsize=(6, 4))
    for dx in [0.1]:
        simulator.Chi = 140
        simulator.Cm = 0.01
        simulator.theta = 1
        simulator.load_mesh(dx=dx)
        simulator.add_material_property(material_config)
        simulator.assemble()
        simulator.solve(a_tol=1e-5, 
                        r_tol=1e-5, 
                        max_iter=1000, 
                        plot_interval=dt * 10, 
                        verbose=True)
    
        if simulator.dx == 0.1:
            color = 'red'
        elif simulator.dx == 0.2:
            color = 'green'
        elif simulator.dx == 0.5:
            color = 'blue'

        plt.plot(simulator.diagonal_distance.cpu().numpy().tolist(), 
                 simulator.activation_time.cpu().numpy().tolist(), 
                 color=color,
                 label=f'dx = {simulator.dx}')


    # Labels and title
    plt.xlabel("Distance (mm)", fontsize=14)
    plt.ylabel("Activation time (ms)", fontsize=14)

    plt.xticks([0, 5, 10, 16, 21.4], fontsize=14)
    plt.yticks([10, 20, 30, 40, 50], fontsize=14)
    
    plt.legend(fontsize=14) 
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("activation_time.pdf", format="pdf")