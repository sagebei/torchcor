import torch
import torchcor as tc

@torch.jit.script
class AlievPanfilov:
    """
    Correct PyTorch implementation of:
      Rubin R. Aliev & Alexander V. Panfilov (1996)
      "A simple two-variable model of cardiac excitation"
      Chaos, Solitons & Fractals, 7(3), 293–301.

    Implements:
      Vm: dimensional membrane potential (mV)
      V : recovery variable (dimensionless)

      U = (Vm - vm_rest) / vm_norm

      diff_V  = -(eps + mu1 * V/(mu2 + U)) * ( V + K*U*(U - a - 1) ) / t_norm
      Iion    = ( K*U*(U - a)*(U - 1) + U*V ) * touAcm2
    """

    def __init__(
        self,
        dt: float,
        device: torch.device=tc.get_device(),
        dtype: torch.dtype = torch.float32
    ):
        self.name = "AlievPanfilov"
        self.dt = dt
        self.device = tc.get_device()
        self.dtype = dtype
        

        # ----- Model constants (from specification) -----
        self.t_norm  = 12.9
        self.vm_norm = 100.0
        self.vm_rest = -80.0
        self.touAcm2 = 100.0 / 12.9    # conversion factor

        # Model parameters
        self.K = 8.0
        self.epsilon = 0.002
        self.a = 0.15
        self.mu1 = 0.2
        self.mu2 = 0.3

        # State variables (vectors)
        self.Vrec = torch.tensor(0.0, device=device, dtype=dtype)  # recovery variable

    # ----------------------------------------------------
    # Initialisation
    # ----------------------------------------------------
    def initialize(self, n_nodes: int):
        """
        Vm initialised to Vm_rest (U=0)
        Recovery variable V initialised to 0
        """
        Vm0 = torch.full((n_nodes,), self.vm_rest, device=self.device, dtype=self.dtype)
        self.Vrec = torch.zeros(n_nodes, device=self.device, dtype=self.dtype)
        return Vm0

    # ----------------------------------------------------
    # Normalised voltage
    # ----------------------------------------------------
    def U_from_Vm(self, Vm: torch.Tensor) -> torch.Tensor:
        return (Vm - self.vm_rest) / self.vm_norm

    # ----------------------------------------------------
    # Time derivatives
    # ----------------------------------------------------
    def differentiate(self, Vm: torch.Tensor) -> torch.Tensor:
        """
        Returns dVm/dt (ionic only)
        and updates recovery variable internally.
        """
        U = self.U_from_Vm(Vm)  # dimensionless

        # ----- Recovery variable derivative dV/dt -----
        dV = -(self.epsilon + self.mu1 * self.Vrec / (self.mu2 + U)) \
             * (self.Vrec + self.K * U * (U - self.a - 1.0)) / self.t_norm

        # Update recovery variable
        self.Vrec = self.Vrec + self.dt * dV

        # ----- Ionic current (dimensional) -----
        Iion = (self.K * U * (U - self.a) * (U - 1.0) + U * self.Vrec) * self.touAcm2

        # Return dVm/dt = -Iion  
        return -Iion



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    dt = 0.02
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ionic = AlievPanfilov(
        dt=dt,
        device=device,
        dtype=torch.float64
    )

    Vm = ionic.initialize(n_nodes=1)

    V_list = []
    t = 0.0
    stim_amp = 40.0  # example: mV/ms units for applied stimulus

    for _ in range(int(400/dt)):
        V_list.append([t, Vm.item()])

        dVm = ionic.differentiate(Vm)
        Vm = Vm + dt * (dVm)

        # external stimulus for first 2 ms
        if 0 <= t <= 2.0:
            Vm = Vm + dt * stim_amp

        t += dt

    V_list = np.array(V_list)

    plt.figure()
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.xlabel("Time (ms)")
    plt.ylabel("Vm (mV)")
    plt.title("Aliev–Panfilov Action Potential")
    plt.grid()
    plt.savefig("aliev_panfilov.png")
