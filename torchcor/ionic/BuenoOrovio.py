import torch
import torchcor as tc
from math import exp, log, sqrt, tanh
from typing import Optional, List


@torch.jit.script
class BuenoOrovio:
    def __init__(self, 
                 dt: float, 
                 region_ids: Optional[List[int]] = None, 
                 formulation: str = "TRANSFORM", 
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64):
        
        self.name = "BuenoOrovio"
        self.dt = dt
        self.region_ids = region_ids
        self.node_indices = torch.tensor([0])
        self.device = device
        self.dtype = dtype

        self.formulation = "TRANSFORM" if formulation is None else formulation

        # Constants
        self.Cm = 1.0
        self.ORIGINAL = 1.
        self.TRANSFORM = 0.
        self.s_init = 0.
        self.theta_w_1 = 0.13
        self.theta_w_2 = 0.13
        self.theta_w_3 = 0.13
        self.theta_w_j_minus = 0.13
        self.theta_w_j_plus = 0.13
        self.v_init = 1.
        self.w_init = 1.

        # Parameters
        self.V_0 = -84.
        self.V_mu = 85.7
        self.k_s = 2.0994
        self.k_so = 2.0
        self.k_w_minus = 200.0
        self.modelformulation = self.ORIGINAL if formulation == "ORIGINAL" else self.TRANSFORM
        self.tau_fi = 0.1
        self.tau_o1 = 470.0
        self.tau_o2 = 6.0
        self.tau_s1 = 2.7342
        self.tau_s2 = 2.0
        self.tau_si = 2.9013
        self.tau_so1 = 40.0
        self.tau_so2 = 1.2
        self.tau_v1_minus = 75.0
        self.tau_v2_minus = 10.0
        self.tau_v_plus = 1.4506
        self.tau_w1_minus = 6.0
        self.tau_w2_minus = 140.0
        self.tau_w_inf = 0.0273
        self.tau_w_plus = 280.0
        self.theta_o = 0.006
        self.theta_v = 0.3
        self.theta_v_minus = 0.2
        self.theta_w = 0.13
        self.u_o = 0.0
        self.u_s = 0.9087
        self.u_so = 0.65
        self.u_u = 1.56
        self.u_w_minus = 0.016
        self.w_inf_star = 0.78
        self.V_init = self.V_0 if self.modelformulation == self.TRANSFORM else 0.

        # 0 lookup tables

        # 3 states variables
        self.s = torch.tensor([self.s_init])
        self.v = torch.tensor([self.v_init])
        self.w = torch.tensor([self.w_init])

        if not torch.jit.is_scripting():
            self.differentiate = torch.compile(
                self.differentiate,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )


    def interpolate(self, X, table, mn: float, mx: float, res: float, step: float, mx_idx: int):
        X = torch.clamp(X, mn, mx)
        idx = ((X - mn) * step).to(torch.long)
        lower_idx = torch.clamp(idx, 0, mx_idx - 1)
        higher_idx = lower_idx + 1
        lower_pos = lower_idx * res + mn
        w = ((X - lower_pos) / res).unsqueeze(1)
        return (1 - w) * table[lower_idx] + w * table[higher_idx]

    def construct_tables(self):
        # BuenoOrovio declares no lookup tables, every rate is evaluated on the fly
        pass

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)

        self.s = torch.full((n_nodes,), self.s_init, device=self.device, dtype=self.dtype)
        self.v = torch.full((n_nodes,), self.v_init, device=self.device, dtype=self.dtype)
        self.w = torch.full((n_nodes,), self.w_init, device=self.device, dtype=self.dtype)

        return V

    def differentiate(self, V):
        # Compute storevars and external modvars
        u = (((V-(self.V_0))/self.V_mu) if (self.modelformulation==self.TRANSFORM) else V)
        Jfi = (((((-self.v)*(((u-(self.theta_v)) >= 0.).to(self.dtype)))*(u-(self.theta_v)))*(self.u_u-(u)))/self.tau_fi)
        Jsi = ((((-(((u-(self.theta_w_3)) >= 0.).to(self.dtype)))*self.w)*self.s)/self.tau_si)
        tau_o = ((self.tau_o1*(1.-((((u-(self.theta_o)) >= 0.).to(self.dtype)))))+((((u-(self.theta_o)) >= 0.).to(self.dtype))*self.tau_o2))
        tau_so = (self.tau_so1+(((self.tau_so2-(self.tau_so1))*(1.+(torch.tanh((self.k_so*(u-(self.u_so)))))))/2.))
        Jso1 = (((u-(self.u_o))*(1.-((((u-(self.theta_w_j_minus)) >= 0.).to(self.dtype)))))/tau_o)
        Jso2 = ((((u-(self.theta_w_j_plus)) >= 0.).to(self.dtype))/tau_so)
        Jso = (Jso1+Jso2)
        Iion = (((Jfi+Jso)+Jsi)*(self.V_mu if (self.modelformulation==self.TRANSFORM) else 1.0))

        # Complete Forward Euler Update
        tau_s = (((1.-((((u-(self.theta_w)) >= 0.).to(self.dtype))))*self.tau_s1)+((((u-(self.theta_w)) >= 0.).to(self.dtype))*self.tau_s2))
        tau_v_minus = (((1.-((((u-(self.theta_v_minus)) >= 0.).to(self.dtype))))*self.tau_v1_minus)+((((u-(self.theta_v_minus)) >= 0.).to(self.dtype))*self.tau_v2_minus))
        tau_w_minus = (self.tau_w1_minus+(((self.tau_w2_minus-(self.tau_w1_minus))*(1.+(torch.tanh((self.k_w_minus*(u-(self.u_w_minus)))))))/2.))
        v_inf = (torch.where((u<self.theta_v_minus), 1.0, 0.0))
        w_inf = (((1.-((((u-(self.theta_o)) >= 0.).to(self.dtype))))*(1.-((u/self.tau_w_inf))))+((((u-(self.theta_o)) >= 0.).to(self.dtype))*self.w_inf_star))
        diff_s = ((((1.+(torch.tanh((self.k_s*(u-(self.u_s))))))/2.)-(self.s))/tau_s)
        diff_v = ((((1.-((((u-(self.theta_v)) >= 0.).to(self.dtype))))*(v_inf-(self.v)))/tau_v_minus)-((((((u-(self.theta_v)) >= 0.).to(self.dtype))*self.v)/self.tau_v_plus)))
        diff_w = ((((1.-((((u-(self.theta_w_1)) >= 0.).to(self.dtype))))*(w_inf-(self.w)))/tau_w_minus)-((((((u-(self.theta_w_2)) >= 0.).to(self.dtype))*self.w)/self.tau_w_plus)))
        s_new = self.s+diff_s*self.dt
        v_new = self.v+diff_v*self.dt
        w_new = self.w+diff_w*self.dt

        # Finish the update
        self.s = s_new
        self.v = v_new
        self.w = w_new

        return -Iion


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.01
    dt_imp = float(np.float32(dt))   # limpet keeps the IMP time step in a float
    stimulus = 60
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = BuenoOrovio(formulation="TRANSFORM", 
                       dt=dt_imp, 
                       device=device, 
                       dtype=torch.float64)
    V = ionic.initialize(n_nodes=1)

    V_list = []

    ctime = 0.0
    for _ in range(int(1000/dt)):
        V_list.append([ctime, V.item()])

        if ctime >= 0 and ctime < (0+2.0): 
            V = V + dt * stimulus
        dV = ionic.differentiate(V)
        V = V + dt * dV
        ctime += dt

    plt.figure()
    V_list = np.array(V_list)    
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.savefig("V_BuenoOrovio.png")
