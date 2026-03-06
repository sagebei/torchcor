import torch

# def compute_activation_map(
#     Vm: torch.Tensor,
#     snapshot_interval: float = 1,
#     threshold: float = 200,
# ) -> torch.Tensor:
#     T, N = Vm.shape
#     above = Vm > threshold                                   # (T, N)
#     crossings = above[1:].float() - above[:-1].float()      # (T-1, N)
#     ascending = crossings > 0                                # (T-1, N)
#     has_crossing = ascending.any(dim=0)                      # (N,)
#     first_crossing = torch.argmax(ascending.long(), dim=0)   # (N,)
#     activation_times = first_crossing.float() * snapshot_interval

#     activation_times[~has_crossing] = float('nan')

#     return activation_times


def compute_activation_map(Vm: torch.Tensor, snapshot_interval: float, threshold=10) -> torch.Tensor:
    dVdt = torch.diff(Vm, dim=0) / snapshot_interval
    dVdt = dVdt * (dVdt > 0)
    peak_idx = torch.argmax(dVdt, dim=0) 
    activation_times = peak_idx.float() * snapshot_interval
    max_dVdt, _ = dVdt.max(dim=0)                     
    activation_times[max_dVdt < threshold] = float('nan')

    return activation_times

def compute_repolarization_map(
    Vm: torch.Tensor,
    snapshot_interval: float = 1,
    threshold: float = -70.0,
    search_after: torch.Tensor = None,
) -> torch.Tensor:
    T, N = Vm.shape

    above = Vm > threshold  # (T, N)

    crossings = above[:-1].float() - above[1:].float()  # (T-1, N)
    descending = crossings > 0                           # (T-1, N)

    if search_after is not None:
        act_idx = (search_after / snapshot_interval).long()  # (N,)
        time_idx = torch.arange(T - 1, device=Vm.device).unsqueeze(1)  # (T-1, 1)
        act_idx_clamped = act_idx.clamp(0, T - 2).unsqueeze(0)         # (1, N)
        descending = descending & (time_idx >= act_idx_clamped)

    descending_flipped = descending.flip(0)
    last_crossing_flipped = torch.argmax(descending_flipped.long(), dim=0)  # (N,)
    last_crossing = (T - 2) - last_crossing_flipped                         # (N,)

    repolarization_times = last_crossing.float() * snapshot_interval

    has_crossing = descending.any(dim=0)  # (N,)
    repolarization_times[~has_crossing] = float('nan')

    return repolarization_times


def main():
    Vm = torch.load("./linear/Vm.pt", map_location="cpu")
    snapshot_interval = 5

    if not isinstance(Vm, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(Vm)}")

    print(f"Vm shape : {Vm.shape} (T={Vm.shape[0]}, N={Vm.shape[1]})")
    print(f"Vm range : [{Vm.min():.2f}, {Vm.max():.2f}] mV")
    print(f"dt       : {snapshot_interval} ms → total duration = {(Vm.shape[0]-1)*snapshot_interval:.1f} ms")

    print("Computing activation times...")
    activation_time = compute_activation_map(Vm, snapshot_interval=snapshot_interval)
    print(f"  AT range: [{activation_time.min():.1f}, {activation_time.max():.1f}] ms")

    print("Computing repolarization times...")
    repolarization_time = compute_repolarization_map(
        Vm,
        snapshot_interval=snapshot_interval,
        search_after=activation_time,   # restrict search to post-activation window
    )
    print(f"  RT range: [{repolarization_time.min():.1f}, {repolarization_time.max():.1f}] ms")


if __name__ == "__main__":
    main()



