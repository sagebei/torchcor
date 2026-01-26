import numpy as np
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class StimConfig:
    vtx_filepath: str
    stimulus: torch.Tensor
    start: float
    duration: float
    intensity: float
    period: float
    count: float


class Stimuli:
    def __init__(self, n_nodes, device, dtype):
        self.stimulus_list = []

        self.n_nodes = n_nodes
        self.device = device
        self.dtype = dtype

        self.Istim = torch.zeros((n_nodes,), device=device, dtype=dtype)

    def load_stimulus_region(self, vtx_filepath):
        with Path(vtx_filepath).open("r") as f:
            region_size = int(f.readline().strip())
    
        region = np.loadtxt(vtx_filepath, dtype=int, skiprows=2)
        
        if len(region) != region_size:
            raise Exception(f"Error loading {vtx_filepath}")
        
        return torch.from_numpy(region).to(dtype=torch.long, device=self.device)
    
    def save_stimulus_region(self, region, vtx_filepath):
        region = region.astype(np.int64).reshape(-1)

        with Path(vtx_filepath).open("w") as f:
            f.write(f"{len(region)}\n")
            f.write("extra\n")
            np.savetxt(f, region, fmt="%d")
    
    def add(self, vtx_filepath, start, duration, intensity, period=None, count=1):
        if period is None:
            raise Exception("period is unspecified.")
        
        region = self.load_stimulus_region(vtx_filepath)
        bool_region = torch.zeros((self.n_nodes,), device=self.device, dtype=self.dtype)
        bool_region[region] = 1.0

        sc = StimConfig(vtx_filepath, bool_region * intensity, start, duration, intensity, period, count)
        self.stimulus_list.append(sc)


    def apply(self, t):
        applied_stimulus = []
        for stim in self.stimulus_list:
            p_t = t % stim.period
            t_t = t / stim.period
            if t_t <= stim.count:
                if p_t >= stim.start and p_t <= stim.start + stim.duration:
                    applied_stimulus.append(stim.stimulus)
        
        if len(applied_stimulus) > 0:
            return torch.stack(applied_stimulus).sum(dim=0)
        else:
            return self.Istim


if __name__ == "__main__":
    stim = Stimuli(n_nodes=1000000, device=torch.device("cuda:0"), dtype=torch.float64)
    stim.add(vtx_filepath="/home/bzhou6/Data/atrium/Case_1/Case_1.vtx",
             start=0,
             duration=2,
             intensity=1,
             period=800,
             count=3)
    print(stim.apply(0).sum())