"""Land et al. (2017) active contraction, batched on the GPU.

    Land, Park-Holohan, Smith, dos Remedios, Kentish and Niederer,
    "A model of cardiac contraction based on novel measurements of tension
    development in human cardiomyocytes", J. Mol. Cell. Cardiol. 106 (2017)
    68-83, doi:10.1016/j.yjmcc.2017.03.008.

Six ordinary differential equations (paper eqs. 47-52) driven by cytosolic
calcium and, when the tissue deforms, by fibre stretch and stretch rate.  The
output is active tension (eq. 53).  Everything is a tensor of shape ``(n,)``
over nodes, so one instance covers a whole mesh and never leaves the device.

Units, which are the reference implementation's: calcium micromolar, stretch
dimensionless, stretch rate and rate constants per millisecond, time
milliseconds, tension kilopascal.

Cross-checked term by term against openCARP's ``Stress_Land17.model`` at
revision 7111f6a8ddfc710cd96ce6caff12f89b3f3b6579.  Every parameter and every
rate agrees with Appendix B of the paper.  One equation does not -- see
``length_dependence`` -- and that difference is exposed rather than hidden.
"""

import torch

#: State order.  A single stacked tensor keeps checkpointing and restoring to
#: one slice, which is what mechanics iteration needs.
XS, XW, TRPN, TMBLOCKED, ZETAS, ZETAW = range(6)
N_STATES = 6


class LandParameters:
    """Model constants, defaulting to the paper's whole-organ column.

    ``LandParameters.skinned()`` gives the skinned-myocyte column instead; the
    two differ only in the five values Appendix B lists twice.
    """

    def __init__(self, **overrides):
        # Shared between both columns (Appendix B).
        self.k_TRPN = 0.1        # koff        [1/ms]
        self.n_TRPN = 2.0        # TRPN_n      [-]
        self.TRPN_50 = 0.35      # perm50      [-]
        self.r_s = 0.25          # dr, duty ratio                     [-]
        self.r_w = 0.5           # wfrac                              [-]
        self.gamma_s = 0.0085    # gamma,   distortion decay, S state [-]
        self.gamma_w = 0.615     # gamma_wu, distortion decay, W state[-]
        self.phi = 2.23          # phi                                [-]
        self.A_eff = 25.0        # TOT_A                              [-]
        self.beta_0 = 2.3        # length dependence of tension       [-]
        self.beta_1 = -2.4       # length dependence of calcium sensitivity
        # Whole-organ column.
        self.ca50 = 0.805        # [Ca]T50_ref [uM]
        self.n_Tm = 5.0          # nperm       [-]
        self.k_u = 1.0           # ktm_unblock [1/ms]
        self.k_uw = 0.182        # 0.026 * nu, nu = 7                 [1/ms]
        self.k_ws = 0.012        # 0.004 * mu, mu = 3                 [1/ms]
        self.T_ref = 120.0       # Tref        [kPa]
        for name, value in overrides.items():
            if not hasattr(self, name):
                raise Exception(f"unknown Land parameter {name!r}")
            setattr(self, name, float(value))

    @classmethod
    def skinned(cls, **overrides):
        """The skinned-myocyte column of Appendix B."""
        return cls(**{"ca50": 2.5, "n_Tm": 2.2, "T_ref": 40.5,
                      "k_uw": 0.026, "k_ws": 0.004, **overrides})

    # ----- quantities derived once, paper eqs. 58-63 ----- #
    @property
    def k_wu(self):
        return self.k_uw * (1.0 / self.r_w - 1.0) - self.k_ws          # eq. 59

    @property
    def k_su(self):
        return self.k_ws * (1.0 / self.r_s - 1.0) * self.r_w           # eq. 60

    @property
    def A(self):
        """Instantaneous distortion per unit stretch, A_s = A_w (eq. 58)."""
        return self.A_eff * self.r_s / ((1.0 - self.r_s) * self.r_w + self.r_s)

    @property
    def c_w(self):
        return (self.phi * self.k_uw * (1.0 - self.r_s) * (1.0 - self.r_w)
                / ((1.0 - self.r_s) * self.r_w))                       # eq. 62

    @property
    def c_s(self):
        return (self.phi * self.k_ws * (1.0 - self.r_s) * self.r_w
                / self.r_s)                                            # eq. 63

    @property
    def k_b(self):
        """Tropomyosin blocking rate, eq. 61."""
        return (self.k_u * self.TRPN_50 ** self.n_Tm
                / (1.0 - self.r_s - (1.0 - self.r_s) * self.r_w))


class Land2017:
    """Active tension over ``n`` nodes.

    ``length_dependence`` selects the tension scaling h(lambda) of eq. 32:

    ``"land2017"`` (default) is the paper,
    ``h = max(0, 1 + beta_0 (lam_m + min(lam_m, 0.87) - 1.87))`` with
    ``lam_m = min(lambda, 1.2)`` (eqs. 30-31), which gives h(1) = 1 exactly so
    that tension equals T_ref at maximal calcium, as the paper requires.

    ``"opencarp"`` reproduces ``Stress_Land17.model``, which multiplies by
    ``max(lam_m, 0.87)`` instead.  Its own h is computed but left unused, and
    is written with ``max`` where the paper has ``min``.  The two agree at
    lambda = 1 and diverge either side of it -- at lambda = 1.1 the paper gives
    1.23 against openCARP's 1.10, at lambda = 0.9, 0.77 against 0.90.  Use it
    only to reproduce openCARP.
    """

    def __init__(self, n, device=None, dtype=torch.float64, parameters=None,
                 length_dependence="land2017"):
        if length_dependence not in ("land2017", "opencarp"):
            raise Exception(f"length_dependence must be 'land2017' or "
                            f"'opencarp', got {length_dependence!r}")
        self.n = int(n)
        self.device = device
        self.dtype = dtype
        self.p = parameters if parameters is not None else LandParameters()
        self.length_dependence = length_dependence
        self.state = torch.zeros((N_STATES, self.n), device=device, dtype=dtype)

    # ----- readable views on the stacked state ----- #
    @property
    def XS(self):
        """Post-powerstroke, force-generating crossbridges."""
        return self.state[XS]

    @property
    def XW(self):
        """Pre-powerstroke, weakly bound crossbridges."""
        return self.state[XW]

    @property
    def TRPN(self):
        """Fraction of troponin C with calcium bound."""
        return self.state[TRPN]

    @property
    def blocked(self):
        """Fraction of myosin binding sites blocked by tropomyosin."""
        return self.state[TMBLOCKED]

    @property
    def ZETAS(self):
        """Mean distortion of post-powerstroke crossbridges."""
        return self.state[ZETAS]

    @property
    def ZETAW(self):
        """Mean distortion of pre-powerstroke crossbridges."""
        return self.state[ZETAW]

    # ----- initial conditions ----- #
    def reset(self):
        """All states zero, which is what the reference implementation uses.

        This is not an equilibrium: with no calcium bound and no tropomyosin
        blocking, every binding site is available and crossbridges cycle
        immediately, so tension rises and falls again over the first few
        hundred milliseconds before settling.  ``settle`` avoids that.
        """
        self.state.zero_()
        return self

    def steady_state(self, calcium, stretch=1.0):
        """Set the state to its exact equilibrium at constant calcium and length.

        Solved in closed form rather than integrated.  With the stretch held,
        both distortions vanish and troponin settles at

            TRPN = r / (1 + r),   r = ([Ca]/ca50(lambda)) ** n_TRPN,

        after which the blocked, pre- and post-powerstroke fractions satisfy a
        linear system whose solution is a single expression in the unblocked
        fraction U.

        This replaces integrating to equilibrium, which was not merely slower:
        the blocked-state rate is ``k_b * min(100, TRPN ** (-n_Tm/2)) * U`` and
        starts at 100 times ``k_b``, so for the skinned parameters an 0.1 ms
        step diverged -- 2.648, then -1.723, reaching -149 by the tenth step.
        A closed form has no step to choose and no stability condition.
        """
        p = self.p
        ca = self._broadcast(calcium)
        lam = self._broadcast(stretch)

        ca50 = p.ca50 + p.beta_1 * (torch.clamp(lam, max=1.2) - 1.0)
        r = (ca / ca50).clamp_min(0.0) ** p.n_TRPN
        trpn = r / (1.0 + r)

        half = p.n_Tm / 2.0
        trpn_safe = trpn.clamp_min(1e-30)
        unblock = torch.clamp(trpn_safe ** (-half), max=100.0)
        beta = p.k_b * unblock / (p.k_u * trpn_safe ** half)      # B = beta * U
        a = p.k_uw / (p.k_wu + p.k_ws)                            # W = a * U
        b = p.k_ws * a / p.k_su                                   # S = b * U
        u = 1.0 / (1.0 + beta + a + b)

        self.state = torch.stack([b * u, a * u, trpn, beta * u,
                                  torch.zeros_like(u), torch.zeros_like(u)])
        return self

    def settle(self, calcium, stretch=1.0, dt=0.01, tol=1e-12, max_steps=2_000_000):
        """Integrate to equilibrium.  Prefer :meth:`steady_state`, which is exact.

        Retained to check that closed form against the integrator.  The default
        step is small because the blocked-state rate is stiff from an
        unequilibrated start; ``steady_state`` avoids the question entirely.
        """
        ca = self._broadcast(calcium)
        lam = self._broadcast(stretch)
        zero = torch.zeros_like(lam)
        check = max(1, int(round(1.0 / dt)))          # check ~ once per ms
        for taken in range(1, max_steps + 1):
            previous = self.state.clone()
            self.state = self._advance(self.state, ca, lam, zero, dt)
            if taken % check == 0:
                if (self.state - previous).abs().max().item() < tol * dt:
                    return taken
        raise Exception(f"Land state did not settle in {max_steps} steps at "
                        f"calcium {float(ca.max()):g} uM")

    # ----- the model ----- #
    def _broadcast(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype
                               ).expand(self.n).clone() \
            if not torch.is_tensor(value) or value.ndim == 0 else value

    def _rates(self, state, ca, lam, dlam_dt):
        """Time derivatives of the six states (paper eqs. 47-52)."""
        p = self.p
        s, w, trpn, blocked = state[XS], state[XW], state[TRPN], state[TMBLOCKED]
        zs, zw = state[ZETAS], state[ZETAW]

        lam_m = torch.clamp(lam, max=1.2)
        # Calcium sensitivity shifts with length (eq. 29).
        ca50 = p.ca50 + p.beta_1 * (lam_m - 1.0)
        d_trpn = p.k_TRPN * ((ca / ca50).clamp_min(0.0) ** p.n_TRPN
                             * (1.0 - trpn) - trpn)                    # eq. 47

        # Tropomyosin (eq. 48).  The reference caps TRPN^(-n_Tm/2) at 100 for
        # numerical stability, which matters because TRPN starts at zero.
        half = p.n_Tm / 2.0
        trpn_safe = trpn.clamp_min(1e-30)
        unblock = torch.clamp(trpn_safe ** (-half), max=100.0)
        u = (1.0 - blocked) - w - s                                    # eq. 55
        d_blocked = p.k_b * unblock * u - p.k_u * trpn_safe ** half * blocked

        # Distortion-dependent unbinding (eqs. 56-57).
        gamma_su = p.gamma_s * torch.maximum(zs.clamp_min(0.0),
                                             (-zs - 1.0).clamp_min(0.0))
        gamma_wu = p.gamma_w * zw.abs()

        d_w = p.k_uw * u - p.k_wu * w - p.k_ws * w - gamma_wu * w      # eq. 49
        d_s = p.k_ws * w - p.k_su * s - gamma_su * s                   # eq. 50
        d_zw = p.A * dlam_dt - p.c_w * zw                              # eq. 51
        d_zs = p.A * dlam_dt - p.c_s * zs                              # eq. 52

        return torch.stack([d_s, d_w, d_trpn, d_blocked, d_zs, d_zw])

    def _advance(self, state, ca, lam, dlam_dt, dt):
        """One forward-Euler step, the reference's integrator (``.method(fe)``)."""
        return state + dt * self._rates(state, ca, lam, dlam_dt)

    def _overlap(self, lam):
        """h(lambda), the length dependence of maximal tension (eq. 32)."""
        lam_m = torch.clamp(lam, max=1.2)
        if self.length_dependence == "opencarp":
            return torch.clamp(lam_m, min=0.87)
        h = 1.0 + self.p.beta_0 * (lam_m + torch.clamp(lam_m, max=0.87) - 1.87)
        return h.clamp_min(0.0)                                    # eqs. 30-31

    def _overlap_derivative(self, lam):
        """dh/dlambda of :meth:`_overlap`, exactly.

        ``h = 1 + beta_0 (lam_m + min(lam_m, 0.87) - 1.87)`` with
        ``lam_m = min(lam, 1.2)``, so the slope is ``2 beta_0`` below 0.87,
        ``beta_0`` between 0.87 and 1.2, and zero outside -- above 1.2 because
        ``lam_m`` stops moving, below the clamp because ``h`` is held at zero.
        """
        if self.length_dependence == "opencarp":
            return torch.where((lam > 0.87) & (lam < 1.2),
                               torch.ones_like(lam), torch.zeros_like(lam))
        lam_m = torch.clamp(lam, max=1.2)
        slope = torch.where(lam_m < 0.87, 2.0 * self.p.beta_0,
                            torch.full_like(lam, self.p.beta_0))
        slope = torch.where(lam < 1.2, slope, torch.zeros_like(lam))
        return torch.where(self._overlap(lam) > 0.0, slope, torch.zeros_like(lam))

    def active_stiffness(self, stretch=1.0, state=None):
        """Active stiffness ``Ka = d(dTa/dt)/d(dlambda/dt)``, in kPa.

        Regazzoni and Quarteroni, CMAME 373 (2021) 113506, eq. (44):

            Ka = grad_r g . dh/dlambda_dot  +  dg/dlambda,

        for a tension written as ``Ta = g(r, lambda)`` with states ``r`` obeying
        ``r_dot = h(r, lambda, lambda_dot)``.  For this model the stretch rate
        enters only the two distortion equations, with coefficient ``A`` each,
        and ``g`` depends on the stretch only through the overlap ``h(lambda)``:

            Ka = h(lambda) T_ref/r_s (A_s XS + A_w XW)          (their eq. 50)
                 + h'(lambda) T_ref/r_s ((1 + zeta_s) XS + zeta_w XW).

        The first term is the distortion response; the second is the explicit
        length dependence, which eq. (44) keeps and eq. (42) drops.  Each
        derivative appears exactly once.  Verified against a difference
        quotient of the tension rate in ``benchmark.verification``.
        """
        state = self.state if state is None else state
        lam = self._broadcast(stretch)
        scale = self.p.T_ref / self.p.r_s
        distortion = self._overlap(lam) * scale * self.p.A * (state[XS] + state[XW])
        explicit = self._overlap_derivative(lam) * scale * (
            state[XS] * (state[ZETAS] + 1.0) + state[XW] * state[ZETAW])
        return distortion + explicit

    def tension_rate(self, calcium, stretch, dlam_dt, state=None):
        """``dTa/dt`` along the model's own dynamics, for checking ``Ka``.

        The tension is ``Ta = g(r(t), lambda(t))``, so its rate has *two*
        parts, and both are needed or differentiating this with respect to the
        stretch rate returns eq. (50) rather than eq. (44):

            dTa/dt = grad_r g . r_dot  +  dg/dlambda . lambda_dot.

        Both gradients are difference quotients here, so the check stays
        independent of the analytic ``h'`` that :meth:`active_stiffness` uses.
        """
        state = self.state if state is None else state
        ca = self._broadcast(calcium)
        lam = self._broadcast(stretch)
        rate = self._broadcast(dlam_dt)
        rates = self._rates(state, ca, lam, rate)
        step = 1e-7
        gradient = torch.zeros_like(state)
        for i in range(N_STATES):
            bumped = state.clone(); bumped[i] = bumped[i] + step
            lowered = state.clone(); lowered[i] = lowered[i] - step
            gradient[i] = (self.tension(lam, bumped) - self.tension(lam, lowered))/(2*step)
        explicit = (self.tension(lam + step, state)
                    - self.tension(lam - step, state))/(2.0*step)
        return (gradient * rates).sum(dim=0) + explicit * rate

    def tension(self, stretch=1.0, state=None):
        """Active tension in kPa (eq. 53), without advancing anything."""
        state = self.state if state is None else state
        lam = self._broadcast(stretch)
        return (self._overlap(lam) * (self.p.T_ref / self.p.r_s)
                * (state[XS] * (state[ZETAS] + 1.0) + state[XW] * state[ZETAW]))

    # ----- stepping ----- #
    def trial(self, calcium, stretch, dlam_dt, dt):
        """Tension after one step, leaving the committed state untouched.

        Mechanics iterates within a time step, so tension has to be evaluable
        repeatedly at trial stretches without the model advancing each time.
        """
        state = self._advance(self.state, self._broadcast(calcium),
                              self._broadcast(stretch),
                              self._broadcast(dlam_dt), dt)
        return self.tension(stretch, state=state)

    def step(self, calcium, stretch, dlam_dt, dt):
        """Advance one step and return the tension of the new state."""
        self.state = self._advance(self.state, self._broadcast(calcium),
                                   self._broadcast(stretch),
                                   self._broadcast(dlam_dt), dt)
        return self.tension(stretch)

    # ----- checkpointing, for a mechanics step that has to be retried ----- #
    def checkpoint(self):
        return self.state.clone()

    def restore(self, state):
        self.state.copy_(state)
        return self
