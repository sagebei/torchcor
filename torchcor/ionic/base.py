import torch
from collections import OrderedDict


class BaseCellModel:
    def __init__(self, cell_model, device=None, dtype=torch.float64):
        self.cell_model = cell_model
        self.dt = None
        self.device = tc.get_device() if device is None else device
        self.dtype = dtype

        init_states, init_constants = self.cell_model.initConsts()
        self.states = torch.tensor(init_states, device=device, dtype=dtype)
        self.constants = None

        self.name_constant_dict = OrderedDict()
        (legend_states, legend_algebraics, _, legend_constants) = self.cell_model.createLegends()
        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            self.name_constant_dict[constant_name] = init_constant

        self.H = None
        

    def default_constants(self):
        return dict(self.name_constant_dict)

    def reset_constant(self, name, value):
        self.name_constant_dict[name] = value

    def initialize(self, n_nodes, dt):
        self.states = self.states.repeat(n_nodes, 1).clone()

        self.constants = torch.tensor(list(self.name_constant_dict.values()), device=self.device, dtype=self.dtype)
        U = self.states[:, 0].clone()
        self.H = self.states[:, 1:].clone()
        self.dt = dt

        return U

    def differentiate(self, U):
        self.states[:, 0] = U
        self.states[:, 1:] = self.H

        rates = self.compute_rates(states=self.states, constants=self.constants)

        dH = rates[:, 1:]
        self.H += self.dt * dH

        dU = rates[:, 0]
        return dU

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], self.cell_model.sizeAlgebraic), device=self.device, dtype=self.dtype)

        return rates

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def get_attribute(self, name: str):
        return getattr(self, name, None)


class BaseCellModelRL:
    def __init__(self, cell_model, device, dtype=torch.float64):
        self.cell_model = cell_model
        self.device = device
        self.dtype = dtype

        self.sizeAlgebraic = cell_model.sizeAlgebraic
        self.sizeStates = cell_model.sizeStates
        self.sizeConstants = cell_model.sizeConstants

        init_states, init_constants = self.cell_model.initConsts()
        print(init_states)
        self.states = torch.tensor(init_states, device=device, dtype=dtype)
        self.constants = None

        (legend_states, legend_algebraics, _, legend_constants) = self.cell_model.createLegends()

        self.name_constant_dict = OrderedDict()
        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            if "stim" in constant_name:
                self.name_constant_dict[constant_name] = 0
            else:
                self.name_constant_dict[constant_name] = init_constant

        gate_dict = {}
        for i, legend_state in enumerate(legend_states):
            legend_state = legend_state.lower()
            if "gate" in legend_state:
                state_name = legend_state.split()[0]
                gate_dict[state_name] = i
        self.gate_indices = list(gate_dict.values())
        self.non_gate_indices = [i for i in range(1, self.sizeStates) if i not in self.gate_indices]

        tau_dict = {}
        inf_dict = {}
        for i, legend_algebraic in enumerate(legend_algebraics):
            legend_algebraic = legend_algebraic.lower()
            algebraic_name = legend_algebraic.split()[0]
            if algebraic_name.startswith("tau"):
                tau_dict[algebraic_name.split("_")[1]] = i
            if algebraic_name.endswith("inf") or algebraic_name.endswith("infinity"):
                inf_dict[algebraic_name.split("_")[0]] = i

        # gating_variables = set(tau_dict.keys()).intersection(set(inf_dict.keys()))

        tau_dict = {key: tau_dict[key] for key in list(gate_dict.keys())}
        inf_dict = {key: inf_dict[key] for key in list(gate_dict.keys())}
        self.tau_indices = list(tau_dict.values())
        self.inf_indices = list(inf_dict.values())

        self.H = None
        self.dt = None

    def initialize(self, n_nodes, dt):
        self.states = self.states.repeat(n_nodes, 1).clone()

        self.constants = torch.tensor(list(self.name_constant_dict.values()), device=self.device, dtype=self.dtype)
        U = self.states[:, 0].clone()
        self.H = self.states[:, 1:].clone()
        self.dt = dt

        return U

    def differentiate(self, U):
        self.states[:, 0] = U

        rates, algebraic = self.compute_rates(states=self.states, constants=self.constants)

        # update states
        self.apply_rush_larsen(algebraic, self.dt)
        self.states[:, self.non_gate_indices] += self.dt * rates[:, self.non_gate_indices]

        dU = rates[:, 0]
        return dU

    # def differentiate(self, U):
    #     self.states[:, 0] = U
    #     self.states[:, 1:] = self.H
    #
    #     rates = self.compute_rates(states=self.states, constants=self.constants)
    #     self.dt_ionic = self.dt / 100
    #     for _ in range(100):
    #         dH = rates[:, 1:]
    #         self.H += self.dt_ionic * dH
    #         self.states[:, 1:] = self.H
    #         rates = self.compute_rates(states=self.states, constants=self.constants)
    #
    #     dU = rates[:, 0]
    #
    #     return dU

    def apply_rush_larsen(self, algebraic, dt):
        steady_states = algebraic[:, self.inf_indices]
        time_constants = algebraic[:, self.tau_indices]

        # Update gating variables using Rush-Larsen method
        self.states[:, self.gate_indices] = steady_states + (self.states[:, self.gate_indices] - steady_states) * torch.exp(-dt / time_constants)

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], self.cell_model.sizeAlgebraic), device=self.device, dtype=self.dtype)

        return rates, algebraic

    def default_constants(self):
        return dict(self.name_constant_dict)

    def reset_constant(self, name, value):
        self.name_constant_dict[name] = value

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def get_attribute(self, name: str):
        return getattr(self, name, None)


