import torch
import torchcor as tc

@torch.jit.script
class ModifiedMS2v:
    def __init__(self, dt: float, device: torch.device=tc.get_device(), dtype: torch.dtype = torch.float32):
        self.name = "ModifiedMS2v"
        self.dt = dt
        self.device = tc.get_device()
        self.dtype = dtype

        self.tau_in = 0.1
        self.tau_out = 9.0
        self.tau_open = 100.0
        self.tau_close = 120.0
        self.u_gate = 0.13
        self.u_crit = 0.13

        self.vmin = -80.0
        self.vmax = 20.0
        self.DV  = self.vmax - self.vmin

        self.H = torch.tensor(1.0, device=device, dtype=dtype) 
        
        

    def to_dimensionless(self, U: torch.Tensor) -> torch.Tensor:
        return (U - self.vmin) / self.DV

    def derivative_to_dimensional(self, dU: torch.Tensor) -> torch.Tensor:
        return self.DV * dU

    def initialize(self, n_nodes: int) -> torch.Tensor:
        self.H = torch.full(size=(n_nodes,), fill_value=1.0, device=self.device, dtype=self.dtype)
        u = torch.full(size=(n_nodes,), fill_value=self.vmin, device=self.device, dtype=self.dtype)
        return u

    def differentiate(self, U: torch.Tensor) -> torch.Tensor:
        Uad = self.to_dimensionless(U)
        J_in = -1.0 * self.H * Uad * (Uad - self.u_crit) * (1.0 - Uad) / self.tau_in
        J_out = (1.0 - self.H) * Uad / self.tau_out
        dU = -self.derivative_to_dimensional(J_in + J_out)
        dH = torch.where(Uad > self.u_gate, -self.H / self.tau_close, (1 - self.H) / self.tau_open)
        self.H += self.dt * dH

        return dU



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.02
    stimulus = 50
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = ModifiedMS2v(dt=dt, 
                                device=device, 
                                dtype=torch.float64)
    V = ionic.initialize(n_nodes=1)

    V_list = []
    ctime = 0.0
    for _ in range(int(1000/dt)):
        V_list.append([ctime, V.item()])

        dV = ionic.differentiate(V)
        V = V + dt * dV
        ctime += dt
        if ctime >= 0 and ctime <= (0+2.0): 
            V = V + dt * stimulus
    
    plt.figure()
    V_list = np.array(V_list)    
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.savefig("V.png")