import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import math
import time
import torch
import torchcor as tc
from pathlib import Path
from torchcor.core.mesh import MeshReader, region_node_idx
from torchcor.core.stimulation import Stimuli
from torchcor.electrophysiology.monodomain import Monodomain


# A large but finite "infinity" for not-yet-reached arrival times. Keeping it
# finite avoids nan/inf propagation in the vectorised local solvers below.
_INF = 1.0e9
_INF2 = 5.0e8

# Action-potential foot current, I_foot(s) = A_F/tau_F * exp(s/tau_F) over
# s = t - t_a in [0, T_FOOT], off once Vm reaches V_TH.  These are openCARP's
# dream.Idiff values; set_foot_current overrides them, per ionic region if wanted.
FOOT_AMP = 0.91          # A_F    (mV)
TAU_FOOT = 0.25          # tau_F  (ms)
V_TH = -30.0             # V_th   (mV)
T_FOOT = 5.0             # T_foot (ms)


# --------------------------------------------------------------------------- #
#  Anisotropic eikonal model (fast iterative method)                          #
# --------------------------------------------------------------------------- #
#  The wavefront arrival times t_a are the viscosity solution of the
#  anisotropic eikonal equation (Neic et al. 2017, eq. 24-25)
#
#       sqrt( grad(t_a)^T  V  grad(t_a) ) = 1            in  Omega
#                                   t_a   = t_0          on  Gamma
#
#  with the squared-velocity tensor  V = v_l ll^T + v_t tt^T + v_n nn^T, where
#  v_l = cv_l^2 and v_t = v_n = cv_t^2 are the squared conduction velocities
#  along / across the fibre direction l.  The travel time of a displacement d is
#  ||d||_M = sqrt(d^T M d) with the metric M = V^{-1},
#
#       M = (1/cv_t^2) I + (1/cv_l^2 - 1/cv_t^2) l l^T .
#
#  It is solved with a Jacobi flavour of the Fast Iterative Method: every simplex
#  provides a monotone, consistent local update for each of its vertices and the
#  field is relaxed to convergence.  The local solvers are the standard simplex
#  updates -- a vertex is updated from the opposite facet:
#     * triangle (2-simplex):  vertex <- opposite edge   (1-D minimisation)
#     * tetrahedron (3-simplex): vertex <- opposite face  (2-D minimisation,
#       whose boundary reduces to the edge updates of that face)
#     * line (1-simplex):      vertex <- the other vertex (anisotropic length)
#  Every update is vectorised over the whole mesh so the solve runs on the GPU.
# --------------------------------------------------------------------------- #

# (target, p, q): for each tet vertex, the three edges of its opposite face.
_TET_EDGES = [(0, 1, 2), (0, 2, 3), (0, 3, 1),
              (1, 2, 3), (1, 3, 0), (1, 0, 2),
              (2, 3, 0), (2, 0, 1), (2, 1, 3),
              (3, 0, 1), (3, 1, 2), (3, 2, 0)]
# (target, s0, s1, s2): each tet vertex and its opposite face.
_TET_FACES = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3), (3, 0, 1, 2)]
# (target, p, q): each triangle vertex and its opposite edge.
_TRI_EDGES = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


def build_metric(fibres, sheets, cv_l, cv_t, cv_n, device, dtype):
    """Per-element eikonal metric M = V^{-1} from conduction velocities.

    fibres : (E, 3) unit fibre vectors
    sheets : (E, 3) unit sheet vectors, or None when the mesh has none
    cv_l   : (E,)   conduction velocity along the fibre        (mm/ms)
    cv_t   : (E,)   ... across the fibre, within the sheet     (mm/ms)
    cv_n   : (E,)   ... normal to the sheet                    (mm/ms)

    With a sheet direction the metric is orthotropic -- the three speeds
    openCARP's ``dream.vel_l``, ``vel_t`` and ``vel_n`` describe:

        M = ff^T/cv_l^2 + ss^T/cv_t^2 + nn^T/cv_n^2,    n = f x s.

    It is written below as cv_t everywhere, corrected along f and n.  For an
    orthonormal frame the two are the same thing, since ff^T + ss^T + nn^T = I,
    but this form also says what to do where the frame is missing: a ``.lon``
    file may leave an element's directions as zeros, and there the corrections
    vanish and conduction is isotropic at cv_t.  Written the other way such an
    element would get a singular M -- zero travel cost, infinite speed.  A mesh
    without sheets is the same expression with the sheet-normal term dropped.
    """
    E = fibres.shape[0]
    inv_l2 = (1.0 / (cv_l * cv_l)).view(E, 1, 1)
    inv_t2 = (1.0 / (cv_t * cv_t)).view(E, 1, 1)
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(E, 3, 3)
    ff = fibres.unsqueeze(2) @ fibres.unsqueeze(1)          # (E, 3, 3)
    M = inv_t2 * eye + (inv_l2 - inv_t2) * ff

    if sheets is None:
        if not torch.allclose(cv_t, cv_n):
            raise Exception(
                "vel_n differs from vel_t, but the mesh has no sheet directions "
                "to apply it along: a .lon file must declare two directions per "
                "element for orthotropic conduction. Supply sheets, or leave "
                "vel_n unset so it follows vel_t.")
        return M

    inv_n2 = (1.0 / (cv_n * cv_n)).view(E, 1, 1)
    normal = torch.linalg.cross(fibres, sheets, dim=-1)
    nn = normal.unsqueeze(2) @ normal.unsqueeze(1)
    return M + (inv_n2 - inv_t2) * nn


def _velocity_entry(vel_l, vel_t, vel_n):
    """Validate one region's conduction velocities and return them as a triple.

    ``vel_n`` defaults to ``vel_t``, the transversely isotropic case and the
    only one a mesh without sheet directions can represent.  A non-positive or
    non-finite speed is rejected here rather than at the metric: 1/cv^2 turns a
    negative speed into the same metric as its absolute value, so a sign slip
    would otherwise propagate silently as a plausible activation map.
    """
    values = {"vel_l": vel_l, "vel_t": vel_t,
              "vel_n": vel_t if vel_n is None else vel_n}
    for name, value in values.items():
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise Exception(f"{name} must be a finite positive conduction "
                            f"velocity in mm/ms, got {value}")
        values[name] = value
    return values["vel_l"], values["vel_t"], values["vel_n"]


def region_velocities(regions, velocity_map, device, dtype):
    """Expand a {region_id: (cv_l, cv_t, cv_n)} map onto per-element arrays."""
    cv_l = torch.zeros(regions.shape[0], device=device, dtype=dtype)
    cv_t = torch.zeros(regions.shape[0], device=device, dtype=dtype)
    cv_n = torch.zeros(regions.shape[0], device=device, dtype=dtype)

    covered = torch.zeros(regions.shape[0], device=device, dtype=torch.bool)
    for rid, (vl, vt, vn) in velocity_map.items():
        mask = regions == rid
        cv_l[mask] = vl
        cv_t[mask] = vt
        cv_n[mask] = vn
        covered |= mask

    if not bool(covered.all()):
        missing = torch.unique(regions[~covered]).tolist()
        raise Exception(f"No conduction velocity specified for region(s) {missing}. "
                        f"Call add_velocity(...) for every region.")
    return cv_l, cv_t, cv_n


def _quad(u, metric, v):
    """Batched metric contraction u^T M v for (E,3) vectors and (E,3,3) M."""
    return torch.einsum('ei,eij,ej->e', u, metric, v)


def _edge_terms(nodes, tgt, p, q, metric):
    """Geometry-only constants of the 1-D update of `tgt` from the segment p-q.

    The update minimises, over P = X_p + xi (X_q - X_p), xi in [0, 1],
        T_tgt = T_p + xi (T_q - T_p) + ||X_tgt - P||_M ,
    which only needs a = e^T M e, b = e^T M w, c = w^T M w (e = X_q-X_p,
    w = X_tgt-X_p) and the two endpoint travel times.
    """
    e = nodes[q] - nodes[p]
    w = nodes[tgt] - nodes[p]
    a = _quad(e, metric, e)
    b = _quad(e, metric, w)
    c = _quad(w, metric, w)
    dpt = torch.sqrt(torch.clamp(c, min=0.0))                 # travel time p -> tgt
    dqt = torch.sqrt(torch.clamp(a - 2.0 * b + c, min=0.0))   # travel time q -> tgt
    return torch.stack([tgt, p, q]), torch.stack([a, b, c, dpt, dqt])


def _face_terms(nodes, tgt, s0, s1, s2, metric):
    """Geometry-only constants of the 2-D update of `tgt` from the face s0-s1-s2.

    The interior update minimises, over P = X0 + xi e1 + eta e2 in the face,
        T_tgt = T0 + xi (T1-T0) + eta (T2-T0) + ||X_tgt - P||_M ,
    with e1 = X1-X0, e2 = X2-X0, w = X_tgt-X0.  Stored are the metric Gram
    entries a11,a12,a22, the projections b1,b2, c0 = w^T M w and det.
    """
    e1 = nodes[s1] - nodes[s0]
    e2 = nodes[s2] - nodes[s0]
    w = nodes[tgt] - nodes[s0]
    a11 = _quad(e1, metric, e1)
    a12 = _quad(e1, metric, e2)
    a22 = _quad(e2, metric, e2)
    b1 = _quad(e1, metric, w)
    b2 = _quad(e2, metric, w)
    c0 = _quad(w, metric, w)
    det = a11 * a22 - a12 * a12
    return torch.stack([tgt, s0, s1, s2]), torch.stack([a11, a12, a22, b1, b2, c0, det])


def build_ops(nodes, elems, velocity_map, fibres, sheets, device, dtype):
    """Pre-compute every local-update operator of the mesh, once.

    Returns (edge, face): `edge` drives the 1-D edge/segment updates (triangles,
    tet face-edges, lines), `face` the 2-D tet interior updates.  Each is a pair
    (idx, coef) of stacked per-update index and constant tensors, or None.
    """
    edge_idx, edge_coef, face_idx, face_coef = [], [], [], []
    part = lambda field, idx: None if field is None else field[idx]

    def add_edges(conn, metric, pattern):
        for t, p, q in pattern:
            idx, coef = _edge_terms(nodes, conn[:, t], conn[:, p], conn[:, q], metric)
            edge_idx.append(idx)
            edge_coef.append(coef)

    if elems.Tr.data is not None:
        cv_l, cv_t, cv_n = region_velocities(elems.Tr.region, velocity_map, device, dtype)
        add_edges(elems.Tr.data,
                  build_metric(fibres[elems.Tr.idx], part(sheets, elems.Tr.idx),
                               cv_l, cv_t, cv_n, device, dtype), _TRI_EDGES)

    if elems.Tt.data is not None:
        cv_l, cv_t, cv_n = region_velocities(elems.Tt.region, velocity_map, device, dtype)
        tet = elems.Tt.data
        metric = build_metric(fibres[elems.Tt.idx], part(sheets, elems.Tt.idx),
                              cv_l, cv_t, cv_n, device, dtype)
        add_edges(tet, metric, _TET_EDGES)
        for t, s0, s1, s2 in _TET_FACES:
            idx, coef = _face_terms(nodes, tet[:, t], tet[:, s0], tet[:, s1], tet[:, s2], metric)
            face_idx.append(idx)
            face_coef.append(coef)

    if elems.Ln.data is not None:
        cv_l, _, _ = region_velocities(elems.Ln.region, velocity_map, device, dtype)
        # propagation along a cable is isotropic at cv_l; degenerate source p==q
        metric = build_metric(fibres[elems.Ln.idx], None, cv_l, cv_l, cv_l,
                              device, dtype)
        add_edges(elems.Ln.data, metric, [(0, 1, 1), (1, 0, 0)])

    if not edge_idx and not face_idx:
        raise Exception("No elements found to build the eikonal operator.")

    edge = (torch.cat(edge_idx, dim=1), torch.cat(edge_coef, dim=1)) if edge_idx else None
    face = (torch.cat(face_idx, dim=1), torch.cat(face_coef, dim=1)) if face_idx else None
    return edge, face


def _relax_edges(out, T, edge):
    """Scatter the best 1-D edge update into `out` (one Jacobi pass)."""
    (tgt, p, q), (a, b, c, dpt, dqt) = edge
    tp, tq = T[p], T[q]
    both = (tp < _INF2) & (tq < _INF2)

    # endpoint (single-vertex) candidates
    cand = torch.minimum(torch.where(tp < _INF2, tp + dpt, _INF),
                         torch.where(tq < _INF2, tq + dqt, _INF))

    # interior critical point of the (convex) cost; its stationarity squares to
    #   a(a - u^2) xi^2 - 2 b(a - u^2) xi + (b^2 - u^2 c) = 0
    u = tq - tp
    au = a - u * u
    A = a * au
    B = -2.0 * b * au
    C = b * b - u * u * c
    sq = torch.sqrt(torch.clamp(B * B - 4.0 * A * C, min=0.0))
    safe_A = torch.where(A.abs() > 1e-30, A, 1.0)
    for xi in ((-B + sq) / (2.0 * safe_A),
               (-B - sq) / (2.0 * safe_A),
               b / torch.where(a > 1e-30, a, 1.0)):                  # last: u -> 0
        valid = both & (xi >= 0.0) & (xi <= 1.0)
        nrm = torch.sqrt(torch.clamp(a * xi * xi - 2.0 * b * xi + c, min=0.0))
        cand = torch.minimum(cand, torch.where(valid, tp + xi * u + nrm, _INF))

    out.scatter_reduce_(0, tgt, cand, reduce='amin', include_self=True)


def _relax_faces(out, T, face):
    """Scatter the best 2-D tetrahedral-face interior update into `out`.

    Over the face, the arrival time at the target is

        f(x) = t0 + u.x + sqrt(x.A x - 2 b.x + c),   x = (xi, eta),

    with A the metric Gram matrix of the face edges.  Stationarity gives
    A x - b = -n u for the travel time n = sqrt(x.A x - 2 b.x + c), hence

        x = A^-1 b - n A^-1 u,   n^2 = (c - b.A^-1 b) / (1 - u.A^-1 u),

    which is solved directly.  ``q = u.A^-1 u < 1`` is the causality condition:
    the front cannot cross the face faster than it travels along it.

    This is the standard algebraic constrained-simplex update.  The earlier
    fixed-point iteration restarted a fixed eight-step approximation on every
    global sweep, so the answer depended on element ordering and outer
    convergence did not imply an accurate local minimum.

    Only points strictly inside the face are kept; the boundary is covered by
    the edge updates, which also provide the vertex fallbacks.  Note the
    stationary-point form never divides by a difference of individual source
    times, so it avoids the equal-time special case that the equivalent
    upstream implementation gets wrong.
    """
    (tgt, s0, s1, s2), (a11, a12, a22, b1, b2, c0, det) = face
    t0, t1, t2 = T[s0], T[s1], T[s2]
    u1, u2 = t1 - t0, t2 - t0

    safe_det = torch.where(det > 0.0, det, torch.ones_like(det))
    # A^-1 b, the unconstrained foot of the perpendicular, and A^-1 u.
    x0 = (a22 * b1 - a12 * b2) / safe_det
    y0 = (a11 * b2 - a12 * b1) / safe_det
    dx = (a22 * u1 - a12 * u2) / safe_det
    dy = (a11 * u2 - a12 * u1) / safe_det

    q = u1 * dx + u2 * dy                                  # u.A^-1 u
    h2 = torch.clamp(c0 - b1 * x0 - b2 * y0, min=0.0)      # c - b.A^-1 b
    n = torch.sqrt(h2 / torch.where(q < 1.0, 1.0 - q, torch.ones_like(q)))
    xi, eta = x0 - n * dx, y0 - n * dy

    inside = ((det > 0.0) & (q < 1.0) & (xi >= 0.0) & (eta >= 0.0) & (xi + eta <= 1.0)
              & (t0 < _INF2) & (t1 < _INF2) & (t2 < _INF2))
    cand = torch.where(inside, t0 + u1 * xi + u2 * eta + n, _INF)
    out.scatter_reduce_(0, tgt, cand, reduce='amin', include_self=True)


def fim_eikonal(edge, face, seed_time, tol=1e-3, max_iter=100000, verbose=True):
    """Fast Iterative Method for the anisotropic eikonal equation.

    Sweeps the local updates to convergence and returns the arrival times, with
    ``nan`` at nodes the front never reaches.  Those two ways of not having a
    time are different and are reported differently: a node the front cannot
    reach is geometry -- an isolated island of mesh, or one with no seed -- and
    is a legitimate ``nan``, whereas running out of sweeps means the times that
    *are* finite have not settled yet, and is an error.  Returning a partial map
    silently would let the reaction stage fire cells at arrival times that are
    still moving.
    """
    T = seed_time.clone()
    start = time.time()
    converged = False
    for n_iter in range(1, max_iter + 1):
        out = T.clone()
        if edge is not None:
            _relax_edges(out, T, edge)
        if face is not None:
            _relax_faces(out, T, face)
        out = torch.minimum(out, seed_time)                  # keep seeds pinned
        change = (T - out).abs().max().item()
        T = out
        if change < tol:
            converged = True
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    reached = int((T < _INF2).sum().item())
    if not converged:
        raise Exception(
            f"eikonal did not converge: {max_iter} sweeps left the arrival times "
            f"still moving by {change:.3e} ms, which is above tol = {tol:g} "
            f"({reached}/{T.numel()} nodes reached so far). Raise max_iter, or "
            f"loosen tol if that change is small enough for your purpose.")
    if verbose:
        print(f"eikonal: {n_iter} sweeps | {reached}/{T.numel()} nodes activated | "
              f"{time.time() - start:.2f}s", flush=True)

    return torch.where(T < _INF2, T, torch.full_like(T, float('nan')))


# --------------------------------------------------------------------------- #
#  Stand-alone eikonal simulator                                              #
# --------------------------------------------------------------------------- #
class Eikonal:
    """Anisotropic eikonal activation-time solver with a Monodomain-like API."""

    def __init__(self, device=None, dtype=None):
        self.device = tc.get_device() if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        self.n_nodes = None
        self.nodes = None
        self.elems = None
        self.fibres = None
        self.sheets = None
        self.regions = None
        self.unique_regions = None

        self._vel = {}
        self.seed_time = None
        self.eikonal_AT = None
        self.mesh_path = None

    def load_mesh(self, path="Data/atrium/Case_1", unit_conversion=1000):
        self.mesh_path = Path(path)
        reader = MeshReader(path)
        nodes, elems, regions, fibres = reader.read(unit_conversion=unit_conversion)

        self.n_nodes = nodes.shape[0]
        self.nodes = torch.from_numpy(nodes).to(dtype=self.dtype, device=self.device)
        self.elems = elems.to_torch(self.device)
        self.regions = torch.from_numpy(regions).to(dtype=torch.long, device=self.device)
        self.unique_regions = torch.unique(self.regions).tolist()
        self.fibres = torch.from_numpy(fibres).to(dtype=self.dtype, device=self.device)
        self.sheets = (None if reader.sheets is None else
                       torch.from_numpy(reader.sheets).to(dtype=self.dtype,
                                                          device=self.device))

        self.seed_time = torch.full((self.n_nodes,), _INF, device=self.device, dtype=self.dtype)

    def add_velocity(self, region_ids, vel_l, vel_t, vel_n=None):
        """Conduction velocities (mm/ms, i.e. numerically m/s) per region.

        ``vel_l`` is along the fibre, ``vel_t`` across it within the sheet and
        ``vel_n`` normal to the sheet -- openCARP's ``dream.vel_l/_t/_n``.
        ``vel_n`` defaults to ``vel_t``, which is the transversely isotropic
        case and the only one a mesh without sheet directions can represent.
        """
        if region_ids is None:
            region_ids = self.unique_regions
        entry = _velocity_entry(vel_l, vel_t, vel_n)
        for rid in region_ids:
            self._vel[rid] = entry

    def add_stimulus(self, vtx_filepath, start=0.0):
        """Seed the wavefront from the nodes listed in a .vtx file at time `start`."""
        region = Stimuli(self.n_nodes, self.device, self.dtype).load_stimulus_region(vtx_filepath)
        self.seed_time[region] = torch.minimum(self.seed_time[region],
                                               torch.full_like(self.seed_time[region], float(start)))

    def solve(self, tol=1e-3, max_iter=100000, verbose=True):
        edge, face = build_ops(self.nodes, self.elems, self._vel, self.fibres,
                               self.sheets, self.device, self.dtype)
        self.eikonal_AT = fim_eikonal(edge, face, self.seed_time, tol=tol, max_iter=max_iter, verbose=verbose)
        return self.eikonal_AT


# --------------------------------------------------------------------------- #
#  Reaction-Eikonal simulator                                                 #
# --------------------------------------------------------------------------- #
class ReactionEikonal(Monodomain):
    """Reaction-Eikonal model (Neic et al., J. Comput. Phys. 2017).

    The eikonal model supplies the activation-time field t_a(x); the full ionic
    reaction is then recovered locally by triggering each cell with a current
    centred on its activation time.  Two trigger currents are supported:

      * default: an action-potential foot current  (A_F / tau_F) * exp((t - t_a) / tau_F)
        in the window [t_a, t_a + T_foot], switched off once Vm reaches V_th.
        A_F, tau_F, V_th and T_foot are prescribed, not derived: the defaults are
        openCARP's, which were chosen for its reference setup, and a different
        ionic model may well need different ones.  `set_foot_current` changes
        them, per ionic region if wanted.
      * opt-in via `set_diffusion_current(...)`: a triple-Gaussian diffusion
        current I_diff that approximates div(sigma grad Vm) and must be fit once
        per ionic model.

    The `diffusion` flag selects the model variant:

      * diffusion=False  ->  R-E (eq. 30): NO diffusion term, so every node is an
        independent ODE  Cm dVm/dt = I_foot - I_ion, the foot firing each cell at
        its own t_a.  Cells are electrically decoupled (no current flows between
        neighbours) and there is no linear solve, so it is fast.  Needs only
        add_velocity.  Use it for activation/repolarisation maps and fast runs.
      * diffusion=True   ->  R-E+ (eq. 32): ADDS the monodomain diffusion operator
        (Cm dVm/dt = I_foot - I_ion + div(sigma grad Vm)/beta), solved with
        conjugate gradient every step.  This couples neighbouring
        cells, recovering electrotonic loading / source-sink effects -- needed for
        accurate electrograms / ECGs.  It is slower (a linear solve per step) and
        also needs add_conductivity, calibrated to the same CV as add_velocity.

    Mesh handling, ionic models, FEM assembly, activation / repolarisation maps
    and VTK/IGB export are inherited unchanged from Monodomain, so the interface
    matches the monodomain solver.

    Re-stimulation reuses the seed stimulus' period/count, i.e. each beat repeats
    the same activation sequence; dynamic restitution / reentry is out of scope.

    `eikonal_AT` is the *prescribed* arrival-time field the eikonal model
    solved for -- when each cell is told to fire.  It is not a measured local
    activation time: the voltage-derived LAT, which `activation_map` computes
    from the Vm history, is an output of the reaction stage and will differ from
    the prescribed field by the foot's rise time.  Compare like with like.
    """

    def __init__(self, ionic_models, T, dt, diffusion=False,
                 device=None, dtype=None, mass_lumping=False):
        super().__init__(ionic_models, T, dt, device=device, dtype=dtype, mass_lumping=mass_lumping)

        self.diffusion = diffusion          # False = R-E (no diffusion); True = R-E+ (with diffusion solve)
        self.foot_amp = FOOT_AMP            # foot amplitude A_F (mV)
        self.tau_foot = TAU_FOOT            # foot time constant tau_F (ms)
        self.v_th = V_TH                    # foot cut-off voltage V_th (mV)
        self.t_foot = T_FOOT                # foot window length T_foot (ms)
        self._foot_regions = {}             # region id -> per-region overrides
        self._foot = None                   # parameters expanded for the run

        for im in ionic_models:
            # The ionic models integrate their own gating states with their own
            # dt, and nothing substeps, so a model built with a different dt
            # advances its states the wrong distance every step while its
            # voltage advances by the driver's -- silently, and worst in the
            # gating and calcium variables rather than in Vm.
            if getattr(im, "dt", dt) != dt:
                raise Exception(
                    f"{type(im).__name__} integrates at dt = {im.dt} ms but the "
                    f"simulator steps at dt = {dt} ms; construct the ionic model "
                    f"with the same dt")

        self._vel = {}
        self.sheets = None                  # set by load_mesh when the .lon has them
        self._gauss = None                  # triple-Gaussian diffusion current, if set
        self.eikonal_AT = None
        self._eikonal_solved_tol = None     # tol the cached field was solved to
        self.period = T
        self.count = 1

    def load_mesh(self, path="Data/atrium/Case_1", unit_conversion=1000):
        super().load_mesh(path=path, unit_conversion=unit_conversion)
        self._invalidate_activation()
        reader = MeshReader(path)
        reader.read_fibres()
        self.sheets = (None if reader.sheets is None else
                       torch.from_numpy(reader.sheets).to(dtype=self.dtype,
                                                          device=self.device))

    def add_velocity(self, region_ids, vel_l, vel_t, vel_n=None):
        """Conduction velocities (mm/ms == m/s) per region for the eikonal solve.

        ``vel_l`` is along the fibre, ``vel_t`` across it within the sheet and
        ``vel_n`` normal to the sheet -- openCARP's ``dream.vel_l/_t/_n``.
        ``vel_n`` defaults to ``vel_t``, which is the transversely isotropic
        case and the only one a mesh without sheet directions can represent.
        """
        if region_ids is None:
            region_ids = self.unique_regions
        entry = _velocity_entry(vel_l, vel_t, vel_n)
        for rid in region_ids:
            self._vel[rid] = entry
        self._invalidate_activation()

    def add_stimulus(self, *args, **kwargs):
        """Add a stimulus, discarding any activation field solved without it."""
        super().add_stimulus(*args, **kwargs)
        self._invalidate_activation()

    def _invalidate_activation(self):
        self.eikonal_AT = None
        self._eikonal_solved_tol = None

    def set_foot_current(self, foot_amp=None, tau_foot=None, v_th=None,
                         t_foot=None, region_ids=None):
        override = {k: v for k, v in (("foot_amp", foot_amp), ("tau_foot", tau_foot),
                                      ("v_th", v_th), ("t_foot", t_foot))
                    if v is not None}
        for name, value in override.items():
            if not math.isfinite(float(value)):
                raise Exception(f"{name} must be finite, got {value}")
        if override.get("tau_foot", 1.0) <= 0.0:
            raise Exception(f"tau_foot must be positive, got {override['tau_foot']}")
        if override.get("t_foot", 1.0) <= 0.0:
            raise Exception(f"t_foot must be positive, got {override['t_foot']}")

        if region_ids is None:
            for name, value in override.items():
                setattr(self, name, float(value))
        else:
            for rid in region_ids:
                self._foot_regions.setdefault(rid, {}).update(
                    {k: float(v) for k, v in override.items()})

    def _foot_parameters(self):
        """Foot parameters as per-node tensors, or scalars if no region differs."""
        if not self._foot_regions:
            return self.foot_amp, self.tau_foot, self.v_th, self.t_foot
        full = lambda value: torch.full((self.n_nodes,), float(value),
                                        device=self.device, dtype=self.dtype)
        fields = {name: full(getattr(self, name))
                  for name in ("foot_amp", "tau_foot", "v_th", "t_foot")}
        for rid, override in self._foot_regions.items():
            idx = region_node_idx(self.elems, [rid])
            for name, value in override.items():
                fields[name][idx] = value
        return (fields["foot_amp"], fields["tau_foot"],
                fields["v_th"], fields["t_foot"])

    # ----- coupling to contraction ----- #
    #: Ionic models whose ``Cai`` this reader understands.  Each was checked
    #: against the resting and peak calcium its publication reports: all six
    #: hold micromolar between calls, though they reach that by different
    #: internal scalings, so the list is explicit rather than assumed.
    #: ``model name -> (attribute, factor to micromolar)``.  Most models update
    #: ``Cai`` in place and it is current.  Tomek20 is the exception: it forms
    #: ``Cai`` from the *previous* ``Cai_mM`` and publishes both, so reading
    #: ``Cai`` returns the calcium of one step ago.  Its live state is
    #: ``Cai_mM`` in millimolar, so that is what is read -- an adapter here,
    #: leaving the ionic equations alone.
    CALCIUM_MODELS = {"TenTusscherPanfilov": ("Cai", 1.0),
                      "Tomek19": ("Cai", 1.0),
                      "Tomek20": ("Cai_mM", 1.0e3),
                      "TWorld": ("Cai", 1.0),
                      "OHaraRudy": ("Cai", 1.0),
                      "CourtemancheRamirezNattel": ("Cai", 1.0)}

    def calcium(self, out=None):
        """Cytosolic calcium in micromolar, per node, on the device.

        Written into ``out`` when given, so a caller stepping a contraction
        model can reuse one buffer instead of allocating every step.  Nodes
        claimed by more than one ionic model are resolved the way the voltage
        is: the last model listed owns them.
        """
        if out is None:
            out = torch.empty(self.n_nodes, device=self.device, dtype=self.dtype)
        for im in self.ionic_models:
            name = type(im).__name__
            if name not in self.CALCIUM_MODELS:
                raise Exception(
                    f"{name} carries no cytosolic calcium, so it cannot drive a "
                    f"contraction model; use one of {', '.join(self.CALCIUM_MODELS)}")
            attribute, factor = self.CALCIUM_MODELS[name]
            out[im.node_indices] = getattr(im, attribute).to(dtype=out.dtype) * factor
        return out

    def save_state(self, vm, time=None):
        """The complete state of a run: voltage and every ionic state variable.

        Enough to continue a simulation exactly where it stopped, which is what
        pre-pacing to steady state needs -- the calcium transient of the first
        beat from default initial conditions is not the one a contraction model
        should be driven with.
        """
        state = {"vm": vm.clone(), "time": float(self.T if time is None else time),
                 "ionic": []}
        for im in self.ionic_models:
            n = im.node_indices.numel()
            state["ionic"].append({name: value.clone()
                                   for name, value in vars(im).items()
                                   if torch.is_tensor(value) and value.shape == (n,)})
        return state

    def load_state(self, state):
        """Restore what ``save_state`` captured; returns the voltage."""
        if len(state["ionic"]) != len(self.ionic_models):
            raise Exception(
                f"saved state has {len(state['ionic'])} ionic models but this "
                f"simulator has {len(self.ionic_models)}")
        for im, saved in zip(self.ionic_models, state["ionic"]):
            for name, value in saved.items():
                getattr(im, name).copy_(value)
        return state["vm"].clone()

    def set_diffusion_current(self, alpha, beta, gamma):
        tensor = lambda v: torch.as_tensor(v, device=self.device, dtype=self.dtype)
        alpha, beta, gamma = tensor(alpha), tensor(beta), tensor(gamma)
        if not (alpha.shape == beta.shape == gamma.shape) or alpha.ndim != 1:
            raise Exception("alpha, beta and gamma must be one-dimensional and the "
                            f"same length, got {list(alpha.shape)}, {list(beta.shape)} "
                            f"and {list(gamma.shape)}")
        if not (torch.isfinite(alpha).all() and torch.isfinite(beta).all()
                and torch.isfinite(gamma).all()):
            raise Exception("alpha, beta and gamma must all be finite")
        if bool((gamma <= 0.0).any()):
            raise Exception("every Gaussian width gamma must be positive")
        self._gauss = (alpha, beta, gamma)

    # ----- eikonal stage ----- #
    def eikonal_activation_times(self, tol=1e-3, max_iter=100000, verbose=True):
        seed_time = torch.full((self.n_nodes,), _INF, device=self.device, dtype=self.dtype)
        for stim in self.stimuli.stimulus_list:
            mask = stim.stimulus != 0
            seed_time[mask] = torch.minimum(seed_time[mask],
                                            torch.full_like(seed_time[mask], float(stim.start)))
        if bool((seed_time >= _INF2).all()):
            raise Exception("No stimulus added: the eikonal model has no seed nodes.")

        edge, face = build_ops(self.nodes, self.elems, self._vel, self.fibres,
                               self.sheets, self.device, self.dtype)
        self.eikonal_AT = fim_eikonal(edge, face, seed_time, tol=tol, max_iter=max_iter, verbose=verbose)
        self._eikonal_solved_tol = tol
        return self.eikonal_AT

    # ----- trigger current (couples eikonal -> reaction) ----- #
    def _foot_current(self, t, u, params):
        foot_amp, tau_foot, v_th, t_foot = params
        rel = t - self.eikonal_AT
        beat = torch.clamp(torch.floor(rel / self.period), min=0.0, max=self.count - 1)
        s = rel - beat * self.period                        # time since latest arrival

        if self._gauss is None:
            active = (s >= 0.0) & (s <= t_foot) & (u < v_th)
            rate = (foot_amp / tau_foot) * torch.exp(s / tau_foot)
        else:
            alpha, beta, gamma = self._gauss
            z = (s.unsqueeze(-1) - beta) / gamma            # (N, k)
            rate = (alpha * torch.exp(-z * z)).sum(dim=-1)  # (N,)
            active = (s >= 0.0) & (s <= t_foot)
        return torch.where(active, rate, torch.zeros_like(rate))

    # ----- one time step ----- #
    def step(self, u, t, a_tol, r_tol, max_iter, verbose=False):
        if verbose and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        ### ionic ###
        b = u * self.Cm
        for im in self.ionic_models:
            idx = im.node_indices
            b[idx] = self.Cm * u[idx] + self.dt * im.differentiate(u[idx]) / 100

        ### eikonal-triggered current (also fires the seed nodes) ###
        u_ion = b / self.Cm
        b += self.dt * self._foot_current(t, u_ion, self._foot) / 100

        if verbose and torch.cuda.is_available():
            torch.cuda.synchronize()
        ionic_time = time.time() - start_time
        start_time = time.time()

        ### electric ###
        if self.diffusion:
            u_star = b / self.Cm
            rhs = self.M @ b - (1 - self.theta) * self.dt * self.K @ u_star
            u, n_iter = self.cg.solve(rhs, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        else:
            u = b / self.Cm
            n_iter = 0

        if verbose and torch.cuda.is_available():
            torch.cuda.synchronize()
        electric_time = time.time() - start_time
        return u, n_iter, ionic_time, electric_time

    # ----- driver ----- #
    def _snapshot_stride(self, snapshot_interval):
        """Time steps between saved frames, or an explanation of why there are none.

        The driver can only save on a step boundary, so the interval has to be
        a whole number of steps.  The original truncated -- 0.75 ms at dt = 0.5
        became one step, 0.5 ms -- and any timing derived from the frames was
        then wrong by 50%; an interval below dt truncated to zero and the modulo
        raised ZeroDivisionError.
        """
        stride = snapshot_interval / self.dt
        if not (snapshot_interval > 0.0 and abs(stride - round(stride)) < 1e-9):
            raise Exception(f"snapshot_interval = {snapshot_interval} ms is not a whole "
                            f"number of time steps of dt = {self.dt} ms; frames can "
                            f"only be saved on a step boundary")
        return int(round(stride))

    def _set_pacing(self):
        """The single beat schedule every stimulus has to share.

        Arrival times are propagated once and repeated, so one period and count
        describe the whole run.  Stimuli that disagree cannot be represented:
        the original silently adopted the first one's schedule and ran the rest
        on it.
        """
        schedules = {(stim.period, stim.count) for stim in self.stimuli.stimulus_list}
        if len(schedules) > 1:
            raise Exception(
                f"the reaction-eikonal model repeats one activation sequence, so every "
                f"stimulus must share a period and count; got {sorted(schedules)}. Give "
                f"them a common schedule, or use the monodomain solver for independent "
                f"pacing trains.")
        if schedules:
            self.period, self.count = schedules.pop()

    def solve(self, a_tol=1e-5, r_tol=1e-5, max_iter=100, linear_guess=True,
              snapshot_interval=5, verbose=True, result_path=None,
              eikonal_tol=1e-3, eikonal_max_iter=100000,
              state=None, resume=False, on_step=None, record=True):
        """Run the reaction-eikonal model.

        ``state`` starts from a ``save_state`` snapshot instead of the ionic
        models' default initial conditions.  By default that is a *new run from
        a prepared initial condition* -- the clock starts at zero, which is
        what pre-pacing to a steady state is for.  ``resume=True`` instead
        continues the earlier run, restoring its clock so that trigger timing
        and pacing phase carry over; a split run then matches an unbroken one.

        ``on_step(t, vm, calcium)`` is called once before the first step with
        the initial state at t = 0, and again after every completed step with
        the time that step reached.  Both tensors are reused between calls, so
        a caller that keeps them must clone.  This is how a contraction model
        is driven; nothing about it belongs in this solver.

        ``record=False`` returns only the final voltage rather than the whole
        history, for coupled runs where the history is consumed as it is made.
        """
        self.result_path = Path(result_path) if result_path is not None else None
        self.snapshot_interval = snapshot_interval

        # 1. eikonal activation times.  A field already solved standalone is
        #    reused, but only if it was solved at least as tightly as asked for
        #    now -- a tighter request must not be answered with a looser answer.
        if self.eikonal_AT is None or self._eikonal_solved_tol > eikonal_tol:
            self.eikonal_activation_times(tol=eikonal_tol, max_iter=eikonal_max_iter, verbose=verbose)

        # 2. FEM operators only needed for the R-E+ diffusion term
        if self.diffusion:
            self.assemble()

        # 3. initial state: the ionic models' own, or a saved one continued
        u = torch.zeros((self.n_nodes), dtype=self.dtype, device=self.device)
        for im in self.ionic_models:
            u[im.node_indices] = im.initialize(im.node_indices.shape[0]).clone()
        start_time = 0.0
        if state is not None:
            u = self.load_state(state).to(dtype=self.dtype, device=self.device)
            if resume:
                # True continuation: pick the clock up where it was left, so the
                # trigger current and any pacing keep their phase.  Without this
                # the run restarts at t = 0 and re-fires every arrival.
                start_time = state["time"]
        u_initial = u.clone()

        # 4. foot-current parameters, prescribed (see set_foot_current)
        self._foot = self._foot_parameters()
        self._set_pacing()

        if self.diffusion:
            self.cg.initialize(x=u, linear_guess=linear_guess)
        ts_per_frame = self._snapshot_stride(snapshot_interval)

        t = start_time
        solving_time = time.time()
        total_ionic_time = 0.0
        total_electric_time = 0.0
        n_total_iter = 0
        solution_list = [u_initial] if record else []
        ca_buffer = None
        if on_step is not None:
            ca_buffer = torch.empty(self.n_nodes, device=self.device, dtype=self.dtype)
            on_step(t, u, self.calcium(out=ca_buffer))
        for n in range(1, self.nt + 1):
            # The step is evaluated at the time it starts from, matching the
            # simulation time openCARP hands its trigger current; t advances
            # afterwards so the frame saved below is labelled by its end time.
            u, n_iter, ionic_time, electric_time = self.step(u, t, a_tol, r_tol, max_iter, verbose)
            t += self.dt
            # The shared solver only reports how many iterations it took, and it
            # stops counting once it converges, so reaching the cap means the
            # tolerance was never met.
            if self.diffusion and n_iter >= max_iter:
                raise Exception(f"conjugate gradient did not converge at t = {t:.4f} ms "
                                f"within max_iter = {max_iter} iterations")

            n_total_iter += n_iter
            total_ionic_time += ionic_time
            total_electric_time += electric_time
            if on_step is not None:
                on_step(t, u, self.calcium(out=ca_buffer))

            if record and n % ts_per_frame == 0:
                solution_list.append(u.clone())
                if verbose and snapshot_interval != self.T:
                    print(f"t: {round(t, 1)}/{self.T} |",
                          f"Time elapsed: {round(time.time() - solving_time, 1)} |",
                          f"CG iter:", n_total_iter, flush=True)
                n_total_iter = 0

        if verbose:
            model = "R-E+" if self.diffusion else "R-E"
            print(f"[{model}] nodes: {self.n_nodes} | "
                  f"total_time: {time.time() - solving_time:.2f}s | "
                  f"ionic_time: {total_ionic_time:.2f}s | "
                  f"electric_time: {total_electric_time:.2f}s", flush=True)

        return torch.stack(solution_list, dim=0) if record else u
