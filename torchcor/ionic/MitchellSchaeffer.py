import torch
import torchcor as tc
from math import exp, log, sqrt
from typing import Optional, List


@torch.jit.script
class MitchellSchaeffer:
    def __init__(self, 
                 dt: float, 
                 region_ids: Optional[List[int]] = None, 
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64):
        
        self.name = "MitchellSchaeffer"
        self.dt = dt
        self.region_ids = region_ids
        self.node_indices = torch.tensor([0])
        self.device = device
        self.dtype = dtype

        # Constants
        self.h_init = 1.0

        # Parameters
        self.V_gate = 0.13
        self.V_max = 1.0
        self.V_min = 0.0
        self.a_crit = 0.0
        self.tau_close = 150.0
        self.tau_in = 0.3
        self.tau_open = 120.0
        self.tau_out = 5.0
        self.V_init = self.V_min

        # 0 lookup tables

        # 1 states variables
        self.h = torch.tensor([self.h_init])

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
        # MitchellSchaeffer declares no lookup tables, every rate is evaluated on the fly
        pass

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)

        self.h = torch.full((n_nodes,), self.h_init, device=self.device, dtype=self.dtype)

        return V

    def differentiate(self, V):
        # Compute storevars and external modvars
        Uamp = (self.V_max-(self.V_min))
        Jin = ((((self.h*((V-(self.V_min))/Uamp))*(((V-(self.V_min))/Uamp)-(self.a_crit)))*((self.V_max-(V))/Uamp))/self.tau_in)
        Jout = (-(((V-(self.V_min))/Uamp)/self.tau_out))
        Iion = ((-Uamp)*(Jin+Jout))

        # Complete Forward Euler Update
        U = ((V-(self.V_min))/Uamp)
        diff_h = (torch.where((U<self.V_gate), ((1.-(self.h))/self.tau_open), ((-self.h)/self.tau_close)))
        h_new = self.h+diff_h*self.dt

        # Finish the update
        self.h = h_new

        return -Iion


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.01
    dt_imp = float(np.float32(dt))   # limpet keeps the IMP time step in a float
    stimulus = 0.3
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = MitchellSchaeffer(dt=dt_imp, 
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
    plt.savefig("V_MitchellSchaeffer.png")
