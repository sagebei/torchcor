"""Linear algebra for :mod:`torchcor.mechanics`: Krylov solvers and preconditioners.

Numerical work uses tensors on the matrix's device.  The tangent is sparse CSR;
a dense global stiffness matrix is never formed.  Host decisions are grouped
at iteration checkpoints.

Because these loops run for thousands of iterations per Newton step, they are
written to keep the host out of the loop: work vectors are allocated once per
solve and updated in place, and the convergence test is evaluated on device and
only *read* every :attr:`KrylovSolver.check_interval` iterations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


__all__ = [
    "Preconditioner", "JacobiPreconditioner", "BlockJacobiPreconditioner",
    "KrylovResult", "KrylovSolver", "ConjugateGradient", "BiCGStab",
]


class Preconditioner(ABC):
    """Preconditioner action ``z = M^{-1} r``.

    BiCGStab applies this on the right; preconditioned CG additionally requires
    a symmetric positive-definite preconditioner.

    ``apply`` accepts an optional ``out`` buffer so that the Krylov loops can
    allocate their work vectors once per solve instead of once per iteration.
    """

    @abstractmethod
    def update(self, assembler, values: torch.Tensor) -> "Preconditioner":
        ...

    @abstractmethod
    def apply(self, r: torch.Tensor,
              out: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...


class JacobiPreconditioner(Preconditioner):
    """Diagonal (point-Jacobi) scaling."""

    def __init__(self) -> None:
        self.inv_diag: Optional[torch.Tensor] = None

    def update(self, assembler, values: torch.Tensor) -> "JacobiPreconditioner":
        d = assembler.diagonal(values)
        self.inv_diag = torch.where(d.abs() > 0, d, torch.ones_like(d)).reciprocal()
        return self

    def apply(self, r: torch.Tensor,
              out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.mul(self.inv_diag, r, out=out)


class BlockJacobiPreconditioner(Preconditioner):
    """Exact inverse of the 3x3 nodal blocks.

    Elasticity couples the three components at a node strongly, so inverting
    the nodal block rather than the scalar diagonal cuts the iteration count
    substantially for the cost of a batched 3x3 inverse.

    Singular blocks are replaced by the identity through ``inv_ex`` and a
    ``where``, which keeps the whole update on device: testing "did any block
    fail?" on the host would synchronise once per Newton step for no benefit.
    """

    def __init__(self, regularisation: float = 1e-12) -> None:
        self.regularisation = float(regularisation)
        if not np.isfinite(self.regularisation) or self.regularisation < 0:
            raise ValueError("regularisation must be finite and non-negative")
        self.inv_blocks: Optional[torch.Tensor] = None

    def update(self, assembler, values: torch.Tensor) -> "BlockJacobiPreconditioner":
        B = assembler.nodal_blocks(values)
        eye = torch.eye(3, dtype=B.dtype, device=B.device)
        scale = B.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).clamp_min(1.0)
        B = B + self.regularisation * scale[:, None, None] * eye

        inv, info = torch.linalg.inv_ex(B)
        ok = (info == 0) & torch.isfinite(inv).flatten(1).all(dim=1)
        self.inv_blocks = torch.where(ok[:, None, None], inv, eye).contiguous()
        return self

    def apply(self, r: torch.Tensor,
              out: Optional[torch.Tensor] = None) -> torch.Tensor:
        rv = r.view(-1, 3, 1)
        if out is None:
            return torch.bmm(self.inv_blocks, rv).view(-1)
        torch.bmm(self.inv_blocks, rv, out=out.view(-1, 3, 1))
        return out


@dataclass
class KrylovResult:
    x: torch.Tensor
    iterations: int
    residual: float
    converged: bool
    #: Conjugate gradients met negative curvature and stopped early.  ``x`` is
    #: then the last iterate before it -- not a solution of the system, but a
    #: descent direction for the energy, which is what Newton-CG uses it as.
    truncated: bool = False


class KrylovSolver(ABC):
    """Preconditioned Krylov solver operating on a torch sparse CSR matrix.

    Parameters
    ----------
    rtol, atol:
        Tolerances on the **true** residual ``|b - A x|``.  ``rtol`` is not
        defaulted far below ``1e-8``: driving a Krylov method much lower is not
        reliably attainable, because the recursively updated residual it
        iterates on drifts from the true one.  Newton does not need it either --
        tighter inner solves should be requested only when the outer problem
        needs them.
    check_interval:
        How often the convergence test is *read back to the host*.  The test
        itself is evaluated on device every iteration; only the decision to stop
        needs a synchronisation, and doing that every iteration is what makes a
        GPU Krylov loop latency-bound.  Overshooting by at most
        ``check_interval - 1`` iterations is avoided by freezing converged
        updates on device until the next host inspection.
    """

    def __init__(self, preconditioner: Optional[Preconditioner] = None,
                 rtol: float = 1e-8, atol: float = 1e-14,
                 max_iter: int = 5000, check_interval: int = 10) -> None:
        self.preconditioner = preconditioner or BlockJacobiPreconditioner()
        self.rtol = float(rtol)
        self.atol = float(atol)
        if not (np.isfinite(self.rtol) and self.rtol >= 0
                and np.isfinite(self.atol) and self.atol >= 0):
            raise ValueError("Krylov tolerances must be finite and non-negative")
        if self.rtol == 0 and self.atol == 0:
            raise ValueError("at least one Krylov tolerance must be positive")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer")
        if not isinstance(check_interval, (int, np.integer)) or check_interval < 1:
            raise ValueError("check_interval must be a positive integer")
        self.max_iter = int(max_iter)
        self.check_interval = int(check_interval)

    @abstractmethod
    def solve(self, A: torch.Tensor, b: torch.Tensor,
              x0: Optional[torch.Tensor] = None) -> KrylovResult:
        ...

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _true_residual(A: torch.Tensor, b: torch.Tensor,
                       x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(b - A @ x)

    def _target(self, b: torch.Tensor, scale: float = 1.0) -> float:
        """Stopping threshold, in the units of the scaled system."""
        return max(self.atol / scale, self.rtol * float(torch.linalg.vector_norm(b)))

    @staticmethod
    def _accept(residual: float, target: float) -> bool:
        """Success needs a *finite* residual meeting a *finite* target.

        Without the finiteness test an infinite right-hand side gives
        ``inf <= inf``, which is true, and the solver reports success on a
        system it never touched.
        """
        return np.isfinite(residual) and np.isfinite(target) and residual <= target

    def _scale_of(self, b: torch.Tensor) -> Optional[float]:
        """Magnitude to divide the system by, or ``None`` if it is unsolvable.

        A right-hand side near the overflow threshold has a perfectly
        representable solution but an unrepresentable *norm*, so working on
        ``b / max|b|`` is what makes such a system solvable instead of
        collapsing to ``inf``.  ``A`` is untouched, so conditioning is
        unchanged; only the reported vector and residual are scaled back.
        """
        scale = float(b.abs().amax())
        if not np.isfinite(scale):
            return None                      # non-finite input: no solution
        # Bounding the scale by atol also keeps atol/scale representable for
        # subnormal right-hand sides that already satisfy the absolute test.
        return 1.0 if scale == 0.0 else max(scale, self.atol)

    def _finish(self, A, b, x, iterations, scale, target,
                allow_success=True) -> KrylovResult:
        """Verify the true residual and the returned, rescaled solution."""
        result = x*scale
        residual, finite = torch.stack([
            self._true_residual(A, b, x),
            torch.isfinite(result).all().to(b.dtype)]).tolist()
        physical_residual = residual*scale
        if not finite or not np.isfinite(physical_residual):
            return KrylovResult(torch.zeros_like(x), iterations, float("inf"), False)
        return KrylovResult(result, iterations, physical_residual,
                            allow_success and self._accept(residual, target))

    def _due(self, it: int) -> bool:
        """Whether to read the convergence test back to the host this iteration.

        Every iteration while the count is still small, then periodically: a
        system that converges in a handful of iterations must not be dragged
        through a whole interval of useless ones, while a long solve should not
        stall the host thousands of times.
        """
        return (it <= self.check_interval or it % self.check_interval == 0
                or it == self.max_iter)

    @staticmethod
    def _safe_div(num: torch.Tensor, den: torch.Tensor,
                  one: torch.Tensor) -> torch.Tensor:
        """Guard inactive zero denominators; callers flag actual breakdowns."""
        zero = den == 0
        return torch.where(zero, 0.0, num) / torch.where(zero, one, den)


class ConjugateGradient(KrylovSolver):
    """Preconditioned conjugate gradients for symmetric positive-definite systems.

    CG additionally *requires* definiteness, which symmetrising a tangent does
    not provide: large deformation, compression, active stress or a pressure
    load can make a tangent indefinite.  The curvature ``p . A p`` is
    therefore monitored on device.  When it turns non-positive the solve stops
    and returns the current iterate, marked ``truncated`` -- the Steihaug rule
    of line-search Newton-CG (Nocedal & Wright, *Numerical Optimization*,
    Alg. 7.1; PETSc's ``KSPSTCG``).  Every CG iterate before that point is a
    descent direction for the energy, whereas the *exact* solution of an
    indefinite system is not, so this is the direction a Newton method should
    take there.  If no iterate has been taken yet, the preconditioned residual
    -- steepest descent -- is returned, as the same rule prescribes.

    Convergence is confirmed against the true residual, not the recursively
    updated one.
    """

    def solve(self, A: torch.Tensor, b: torch.Tensor,
              x0: Optional[torch.Tensor] = None) -> KrylovResult:
        scale = self._scale_of(b)
        if scale is None:
            return KrylovResult(torch.zeros_like(b), 0, float("inf"), False)
        if scale != 1.0:
            b = b / scale
            x0 = None if x0 is None else x0 / scale

        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - A @ x
        target = self._target(b, scale)

        res_dev = torch.linalg.vector_norm(r)
        res = float(res_dev)
        if self._accept(res, target):
            return self._finish(A, b, x, 0, scale, target)
        if not np.isfinite(res) or self.max_iter == 0:
            return self._finish(A, b, x, 0, scale, target, allow_success=False)

        # Work vectors, allocated once and updated in place from here on.
        z = self.preconditioner.apply(r)
        p = z.clone()
        checkpoint = x.clone()
        rz = torch.dot(r, z)
        one = torch.ones((), dtype=b.dtype, device=b.device)
        running = torch.ones((), dtype=torch.bool, device=b.device)
        breakdown = torch.zeros_like(running)

        it = 0
        for it in range(1, self.max_iter + 1):
            Ap = A @ p
            pAp = torch.dot(p, Ap)
            bad = ((pAp <= 0) | (rz <= 0)
                   | ~torch.isfinite(pAp) | ~torch.isfinite(rz))
            breakdown.logical_or_(running & bad)
            running.logical_and_(~breakdown)
            alpha = torch.where(running, self._safe_div(rz, pAp, one), 0.0)

            x.addcmul_(p, alpha)
            r.addcmul_(Ap, alpha, value=-1.0)

            res_dev = torch.linalg.vector_norm(r)
            running.logical_and_(res_dev > target)
            if self._due(it):
                # One host transfer carries both the stopping test and the
                # definiteness test.
                res, failed = torch.stack([res_dev, breakdown.to(b.dtype)]).tolist()
                if not np.isfinite(res):
                    return self._finish(A, b, checkpoint, it, scale, target,
                                        allow_success=False)
                if failed:
                    # Newton-CG: the iterate before the bad direction, or
                    # steepest descent if that iterate is still zero.
                    direction = x if bool((x != 0).any()) else \
                        self.preconditioner.apply(b - A @ x)
                    result = self._finish(A, b, direction, it, scale, target,
                                          allow_success=False)
                    result.truncated = bool(torch.isfinite(result.x).all())
                    return result
                if self._accept(res, target):
                    break
                checkpoint.copy_(x)

            self.preconditioner.apply(r, out=z)
            rz_new = torch.dot(r, z)
            p.mul_(self._safe_div(rz_new, rz, one)).add_(z)
            rz = rz_new

        return self._finish(A, b, x, it, scale, target)


class _BiCGStabState:
    """Fixed work buffers shared by eager execution and optional CUDA graphs."""

    def __init__(self, A, preconditioner, block_size, capture_enabled):
        self.capture_enabled = capture_enabled
        self.A = A
        self.preconditioner = preconditioner
        if capture_enabled:
            self.A = torch.sparse_csr_tensor(
                A.crow_indices(), A.col_indices(), A.values().clone(), size=A.shape,
                dtype=A.dtype, device=A.device, check_invariants=False)
            self.preconditioner = copy(preconditioner)
            for name, value in vars(preconditioner).items():
                if isinstance(value, torch.Tensor):
                    setattr(self.preconditioner, name, value.clone())
        self.block_size = block_size
        self.graph = None
        self.vectors = {name: torch.empty(A.shape[0], dtype=A.dtype, device=A.device)
                        for name in ("x", "r", "r0", "p", "v", "y", "z", "s", "t")}
        self.scalars = {name: torch.empty((), dtype=A.dtype, device=A.device)
                        for name in ("rho", "alpha", "omega", "residual", "target", "rho_scale")}
        self.flags = {name: torch.zeros((), dtype=torch.bool, device=A.device)
                      for name in ("running", "breakdown", "rho_breakdown")}
        self.one = torch.ones((), dtype=A.dtype, device=A.device)

    def sync_problem(self, A, preconditioner):
        if not self.capture_enabled:
            self.A, self.preconditioner = A, preconditioner
            return
        self.A.values().copy_(A.values())
        for name, value in vars(preconditioner).items():
            if isinstance(value, torch.Tensor):
                getattr(self.preconditioner, name).copy_(value)

    def reset(self, x, residual, norm, target):
        self.vectors["x"].copy_(x)
        self.vectors["r"].copy_(residual)
        self.scalars["target"].fill_(target)
        self.restart(norm)

    def restart(self, norm):
        v, s, f = self.vectors, self.scalars, self.flags
        v["r0"].copy_(v["r"])
        v["p"].zero_()
        v["v"].zero_()
        for name in ("rho", "alpha", "omega"):
            s[name].fill_(1.0)
        s["residual"].fill_(norm)
        s["rho_scale"].fill_(torch.finfo(self.A.dtype).eps*norm)
        f["running"].fill_(True)
        f["breakdown"].zero_()
        f["rho_breakdown"].zero_()

    def step(self):
        v, s, f = self.vectors, self.scalars, self.flags
        divide = BiCGStab._safe_div
        rho_new = torch.dot(v["r0"], v["r"])
        f["rho_breakdown"].logical_or_(
            f["running"] & (rho_new.abs() <= s["rho_scale"]*s["residual"]))
        f["running"].logical_and_(~f["rho_breakdown"])
        beta = divide(rho_new, s["rho"], self.one)*divide(s["alpha"], s["omega"], self.one)
        beta = torch.where(f["running"], beta, 0.0)
        v["p"].addcmul_(v["v"], s["omega"], value=-1.0).mul_(beta).add_(v["r"])
        s["rho"].copy_(rho_new)
        self.preconditioner.apply(v["p"], out=v["y"])
        torch.mv(self.A, v["y"], out=v["v"])
        denominator = torch.dot(v["r0"], v["v"])
        f["breakdown"].logical_or_(
            f["running"] & ((denominator == 0) | ~torch.isfinite(denominator)))
        f["running"].logical_and_(~f["breakdown"])
        s["alpha"].copy_(torch.where(
            f["running"], divide(s["rho"], denominator, self.one), 0.0))
        v["s"].copy_(v["r"]).addcmul_(v["v"], s["alpha"], value=-1.0)
        v["x"].addcmul_(v["y"], s["alpha"])
        self.preconditioner.apply(v["s"], out=v["z"])
        torch.mv(self.A, v["z"], out=v["t"])
        ts, tt = torch.dot(v["t"], v["s"]), torch.dot(v["t"], v["t"])
        s["omega"].copy_(torch.where(f["running"], divide(ts, tt, self.one), 0.0))
        v["x"].addcmul_(v["z"], s["omega"])
        v["r"].copy_(v["s"]).addcmul_(v["t"], s["omega"], value=-1.0)
        s["residual"].copy_(torch.linalg.vector_norm(v["r"]))
        bad_omega = ((ts == 0) | (tt == 0) | ~torch.isfinite(ts) | ~torch.isfinite(tt))
        f["breakdown"].logical_or_(
            f["running"] & (s["residual"] > s["target"]) & bad_omega)
        f["running"].logical_and_((s["residual"] > s["target"]) & ~f["breakdown"])

    def capture(self):
        buffers = {**self.vectors, **self.scalars, **self.flags}
        saved = {name: tensor.clone() for name, tensor in buffers.items()}
        stream = torch.cuda.Stream(device=self.A.device)
        stream.wait_stream(torch.cuda.current_stream(self.A.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream(self.A.device).wait_stream(stream)
        for name, tensor in buffers.items():
            tensor.copy_(saved[name])
        stream.wait_stream(torch.cuda.current_stream(self.A.device))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            for _ in range(self.block_size):
                self.step()
        for name, tensor in buffers.items():
            tensor.copy_(saved[name])
        torch.cuda.synchronize(self.A.device)


class BiCGStab(KrylovSolver):
    """Right-preconditioned BiCGStab for a general non-symmetric CSR matrix.

    The recurrence follows the SIAM Templates algorithm (Netlib
    https://www.netlib.org/templates/cpp/bicgstab.h).  Loss of shadow-residual
    orthogonality triggers a restart, using rho and its residual norm from the
    same iteration.  A zero alpha denominator or zero omega before convergence
    is a breakdown; the solver returns the last inspected iterate with a true
    residual and a failure flag.  Recursive convergence is always checked
    against the true residual and restarted when necessary.

    ``use_cuda_graph`` batches iterations into CUDA graph replays for the built-in
    Jacobi preconditioners.  Matrices and preconditioner values may change between
    solves; their static graph buffers are updated each time.  Custom
    preconditioners use the same recurrence eagerly.  Capturing is deferred until
    a solve exceeds ``check_interval`` iterations, so short solves avoid setup.
    This changes execution only, not the stopping test or mathematical method.
    """

    max_replacements: int = 8

    def __init__(self, preconditioner: Optional[Preconditioner] = None,
                 rtol: float = 1e-8, atol: float = 1e-14,
                 max_iter: int = 5000, check_interval: int = 10,
                 use_cuda_graph: bool = True) -> None:
        super().__init__(preconditioner, rtol, atol, max_iter, check_interval)
        self.use_cuda_graph = bool(use_cuda_graph)
        self._state = None
        self._signature = None

    def solve(self, A, b, x0=None):
        scale = self._scale_of(b)
        if scale is None:
            return KrylovResult(torch.zeros_like(b), 0, float("inf"), False)
        if scale != 1.0:
            b = b/scale
            x0 = None if x0 is None else x0/scale
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - A @ x
        target = self._target(b, scale)
        norm = float(torch.linalg.vector_norm(r))
        if self._accept(norm, target):
            return self._finish(A, b, x, 0, scale, target)
        if not np.isfinite(norm) or self.max_iter == 0:
            return self._finish(A, b, x, 0, scale, target, allow_success=False)
        capture_enabled = (self.use_cuda_graph and A.is_cuda
                           and type(self.preconditioner) in
                           (JacobiPreconditioner, BlockJacobiPreconditioner))
        buffers = tuple((name, value.shape, value.dtype, value.device)
                        for name, value in vars(self.preconditioner).items()
                        if isinstance(value, torch.Tensor))
        signature = (A.crow_indices().data_ptr(), A.col_indices().data_ptr(),
                     A.shape, A.dtype, A.device, type(self.preconditioner),
                     buffers, self.check_interval, capture_enabled)
        if self._state is None or self._signature != signature:
            self._state = _BiCGStabState(
                A, self.preconditioner, self.check_interval, capture_enabled)
            self._signature = signature
        state = self._state
        state.sync_problem(A, self.preconditioner)
        state.reset(x, r, norm, target)
        v, scalars, flags = state.vectors, state.scalars, state.flags
        checkpoint = x.clone()
        replacements, iterations = 0, 0
        while iterations < self.max_iter:
            if (state.capture_enabled and iterations >= self.check_interval
                    and self.max_iter - iterations >= self.check_interval):
                if state.graph is None:
                    state.capture()
                state.graph.replay()
                iterations += self.check_interval
            else:
                state.step()
                iterations += 1
                if not self._due(iterations):
                    continue
            norm, failed, rho_bad = torch.stack([
                scalars["residual"], flags["breakdown"].to(b.dtype),
                flags["rho_breakdown"].to(b.dtype)]).tolist()
            if not np.isfinite(norm) or failed:
                return self._finish(A, b, checkpoint, iterations, scale, target,
                                    allow_success=False)
            if self._accept(norm, target) or rho_bad:
                r_true = b - A @ v["x"]
                norm = float(torch.linalg.vector_norm(r_true))
                if self._accept(norm, target):
                    break
                if not np.isfinite(norm):
                    return self._finish(A, b, checkpoint, iterations, scale, target,
                                        allow_success=False)
                if not rho_bad:
                    if replacements >= self.max_replacements:
                        break
                    replacements += 1
                checkpoint.copy_(v["x"])
                v["r"].copy_(r_true)
                state.restart(norm)
            else:
                checkpoint.copy_(v["x"])
        return self._finish(A, b, v["x"], iterations, scale, target)
