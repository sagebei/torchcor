import sys
import os
import warnings

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


import torch
from torchcor.core.assemble import Matrices2D
from torchcor.core.preconditioner import Preconditioner
from torchcor.core.solver import ConjugateGradient
from torchcor.core.visualize import VTK2D, GIF2D
from torchcor.core.reorder import RCM as RCM
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from torchcor.core.boundary import apply_dirichlet_boundary_conditions

def compute_w(t, vertices, k=2, omega_1=2 * math.pi, omega_2=math.pi, lbda=math.pi/4):
    a = omega_1 * vertices[:, 0] + omega_2 * vertices[:, 1] - lbda * t
    return math.exp(-k * t) * torch.cos(a)

def compute_r(w, t, vertices, k=2, omega_1=2 * math.pi, omega_2=math.pi, lbda=math.pi/4):
    a = omega_1 * vertices[:, 0] + omega_2 * vertices[:, 1] - lbda * t
    return math.exp(-k * t) * (-k * torch.cos(a) + lbda * torch.sin(a)) + (omega_1 ** 2 + omega_2 ** 2) * w

class Monodomain:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=None):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float64

        self.T = T  # ms 
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
        self.region_ids = None

        self.material_config = None

        self.Chi = 1  # mm
        self.Cm = 1  # uFmm
        self.theta = 1
        self.K = None
        self.M = None
        self.A = None

        self.stimulus_region = None
        self.stimuli = []

        self.dirichlet_boundary_nodes = None

    def load_mesh(self):
        L = 1  # Length of domain in x and y directions

        x = np.linspace(0, L, self.Nx)
        y = np.linspace(0, L, self.Ny)
        X, Y = np.meshgrid(x, y)

        # Flatten the X, Y, Z arrays for input to Delaunay
        vertices = np.vstack([X.flatten(), Y.flatten()]).T

        self.triangles = torch.from_numpy(Delaunay(vertices).simplices).to(dtype=torch.long, device=device)
        self.vertices = torch.from_numpy(vertices).to(dtype=self.dtype, device=device)

        self.n_nodes = self.vertices.shape[0]
        
        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.triangles)

    def add_material_property(self, material_config):
        self.material_config = material_config

    def assemble(self):
        if self.rcm is not None:
            rcm_vertices = self.rcm.reorder(self.vertices)
            rcm_triangles = self.rcm.map(self.triangles)
        else:
            rcm_vertices = self.vertices
            rcm_triangles = self.triangles
        
        sigma = torch.tensor([[1, 0],
                              [0, 1]], device=self.device, dtype=self.dtype)

        matrices = Matrices2D(rcm_vertices, rcm_triangles, device=self.device, dtype=self.dtype)
        K, M = matrices.assemble_matrices(sigma)
        
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M * self.Cm * self.Chi + self.K * self.dt * self.theta

        # the nodes on the 4 borders.
        self.dirichlet_boundary_nodes = torch.where((self.vertices[:, 0] == 0) | 
                                                    (self.vertices[:, 0] == 1) | 
                                                    (self.vertices[:, 1] == 0) | 
                                                    (self.vertices[:, 1] == 1))[0]

        A = apply_dirichlet_boundary_conditions(A, self.dirichlet_boundary_nodes)

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)
        self.A = A.to_sparse_csr()

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u0 = compute_w(t=0, vertices=self.vertices)
        u = u0

        cg = ConjugateGradient(self.pcd, self.A)
        cg.initialize(x=u)

        t = 0
        n_total_iter = 0

        total_diff = 0
        total_ref = 0

        for n in range(1, self.nt + 1):
            t += self.dt

            w = compute_w(t, self.vertices)
            Iion = -compute_r(w, t, self.vertices)

            b = u * self.Cm - self.dt * Iion
            b = self.M @ b * self.Chi
            b -= (1 - self.theta) * self.K @ u * dt
            # boundary condition
            b[self.dirichlet_boundary_nodes] = w[self.dirichlet_boundary_nodes]
            
            u, n_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            if n_iter == max_iter:
                raise Exception(f"The solution did not converge at {n}th timestep")
            if verbose:
                print(f"{round(t, 3)} / {self.T}: {n_iter}")
            n_total_iter += n_iter
            
            diff = torch.norm(u - w, p=2) / torch.norm(w, p=2)
            total_diff += torch.norm(u - w, p=2) ** 2
            total_ref += torch.norm(w, p=2) ** 2

            # diff_list.append([t, diff.item()])
        final_diff = torch.sqrt(total_diff) / torch.sqrt(total_ref)
        print(f"dt={self.dt}", f"dx={1/self.Nx}", final_diff.item())

        # print(f"total iterations: {n_total_iter}")
        # fig, ax = plt.subplots(figsize=(6, 4))
        # diff_list = np.array(diff_list)    
        # plt.plot(diff_list[:, 0][:-200], diff_list[:, 1][:-200])

        # plt.xlabel("Time (ms)", fontsize=14, fontname='Times New Roman')
        # plt.ylabel("Relative norm difference", fontsize=14, fontname='Times New Roman')
        
        # x_space = np.linspace(0, 30, 5).tolist()
        # plt.xticks(x_space)
        # y_space = np.linspace(0, 0.006, 4).tolist()
        # plt.yticks(y_space)
        
        # # plt.legend()
        # plt.grid(True, linestyle="--", alpha=0.5)
        # plt.tight_layout()
        # plt.savefig("maufactured.pdf", format="pdf")

if __name__ == "__main__":
    for dt in [1/20, 1/40, 1/80, 1/160, 1/320]:
        for N in [80, 160, 320]:
            try:
                device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

                simulator = Monodomain(ionic_model=None, 
                                    T=1, 
                                    dt=dt, 
                                    apply_rcm=False, 
                                    device=device)
                simulator.Nx = N
                simulator.Ny = N
                simulator.load_mesh()
                simulator.add_material_property(material_config=None)
                simulator.assemble()
                simulator.solve(a_tol=1e-8, 
                                r_tol=1e-8, 
                                max_iter=1000, 
                                plot_interval=dt * 10, 
                                verbose=False)
            except Exception:
                print(f"dt={dt}", f"dx={1/N}", 0)