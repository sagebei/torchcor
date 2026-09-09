"""VTK output for :mod:`torchcor.mechanics`.

Writes ParaView-readable XML VTK files with **no third-party dependencies** --
no ``vtk``, ``meshio`` or ``pyvista`` needed, which keeps the mechanics module
importable anywhere torch is.

The module is one generic writer plus a few thin wrappers::

    write_vtu       generic unstructured grid (points + one cell type)
    write_mesh      a hexahedral mesh, in its reference configuration
    write_deformed  the same mesh warped by a displacement field
    write_points    marker points
    write_polyline  a curve through a list of points
    write_scene     reference + deformed + markers + curves, in one call
    write_pvd       a ParaView collection, for load-step or time series

Reproducing figure 1 of Land et al. (2015) -- the reference beam below its
deformed solution, with the probe node, the mid-line and the strain points
marked -- is a single :func:`write_scene` call.  The benchmarks themselves
report with :func:`render_deformation`; see
``torchcor.mechanics.benchmark.p1.write_figure``.

High-order cells are written as ``p ** 3`` linear sub-hexahedra rather than as
VTK Lagrange cells: every ParaView version renders them, and the subdivision
resolves the curved solution instead of flattening it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "VTK_VERTEX",
    "VTK_LINE",
    "VTK_POLY_LINE",
    "VTK_HEXAHEDRON",
    "write_vtu",
    "write_mesh",
    "write_deformed",
    "write_points",
    "write_polyline",
    "write_scene",
    "write_pvd",
    "render_deformation",
]

VTK_VERTEX = 1
VTK_LINE = 3
VTK_POLY_LINE = 4
VTK_HEXAHEDRON = 12


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _as_numpy(a, dtype=None) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    return a.astype(dtype) if dtype is not None else a


def _floats(a: np.ndarray) -> str:
    return " ".join(f"{v:.10g}" for v in a.reshape(-1))


def _ints(a: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in np.asarray(a).reshape(-1))


def _data_arrays(data: Optional[Mapping[str, object]], n: int,
                 tag: str) -> list[str]:
    """Serialise a name -> array mapping as a ``<PointData>``/``<CellData>`` block."""
    if not data:
        return []
    lines = [f"      <{tag}>"]
    for name, value in data.items():
        arr = _as_numpy(value, float).reshape(n, -1)
        lines += [
            f'        <DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="{arr.shape[1]}" format="ascii">',
            "          " + _floats(arr),
            "        </DataArray>",
        ]
    lines.append(f"      </{tag}>")
    return lines


# ---------------------------------------------------------------------------
#  Generic writer
# ---------------------------------------------------------------------------

def write_vtu(
    path: str | Path,
    points,
    cells,
    cell_type: int,
    point_data: Optional[Mapping[str, object]] = None,
    cell_data: Optional[Mapping[str, object]] = None,
) -> Path:
    """Write an unstructured grid of a single cell type to a ``.vtu`` file.

    Parameters
    ----------
    points:
        ``(n_points, 3)`` coordinates.
    cells:
        ``(n_cells, nodes_per_cell)`` connectivity into ``points``.
    cell_type:
        VTK cell type id, e.g. :data:`VTK_HEXAHEDRON`.
    point_data, cell_data:
        Optional ``name -> array`` mappings; arrays are reshaped to
        ``(n, n_components)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = _as_numpy(points, float).reshape(-1, 3)
    conn = _as_numpy(cells, np.int64).reshape(len(_as_numpy(cells)), -1)
    n_pts, n_cells, per_cell = pts.shape[0], conn.shape[0], conn.shape[1]

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">',
        "      <Points>",
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
        "          " + _floats(pts),
        "        </DataArray>",
        "      </Points>",
        "      <Cells>",
        '        <DataArray type="Int64" Name="connectivity" format="ascii">',
        "          " + _ints(conn),
        "        </DataArray>",
        '        <DataArray type="Int64" Name="offsets" format="ascii">',
        "          " + _ints(np.arange(1, n_cells + 1) * per_cell),
        "        </DataArray>",
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        "          " + " ".join([str(int(cell_type))] * n_cells),
        "        </DataArray>",
        "      </Cells>",
    ]
    lines += _data_arrays(point_data, n_pts, "PointData")
    lines += _data_arrays(cell_data, n_cells, "CellData")
    lines += ["    </Piece>", "  </UnstructuredGrid>", "</VTKFile>", ""]

    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
#  Mesh wrappers
# ---------------------------------------------------------------------------

def write_mesh(
    path: str | Path,
    mesh,
    point_data: Optional[Mapping[str, object]] = None,
    cell_data: Optional[Mapping[str, object]] = None,
) -> Path:
    """Write a hexahedral mesh in its reference configuration.

    ``mesh`` needs a ``points`` attribute and a ``linear_cells()`` method, i.e.
    any :class:`torchcor.mechanics.mesh.HexMesh`.
    """
    if not hasattr(mesh, "linear_cells"):
        raise TypeError(f"{type(mesh).__name__} has no linear_cells(); expected a HexMesh")
    return write_vtu(path, mesh.points, mesh.linear_cells(), VTK_HEXAHEDRON,
                     point_data=point_data, cell_data=cell_data)


def write_deformed(
    path: str | Path,
    mesh,
    u,
    point_data: Optional[Mapping[str, object]] = None,
    cell_data: Optional[Mapping[str, object]] = None,
) -> Path:
    """Write a mesh warped by the displacement field ``u``.

    ``displacement`` and ``displacement_magnitude`` are attached automatically
    (a supplied ``point_data`` entry of the same name wins), so the result can
    be colour-mapped straight away.
    """
    disp = _as_numpy(u, float).reshape(-1, 3)
    pts = _as_numpy(mesh.points, float).reshape(-1, 3) + disp

    fields: Dict[str, object] = {
        "displacement": disp,
        "displacement_magnitude": np.linalg.norm(disp, axis=1),
    }
    fields.update(point_data or {})

    return write_vtu(path, pts, mesh.linear_cells(), VTK_HEXAHEDRON,
                     point_data=fields, cell_data=cell_data)


# ---------------------------------------------------------------------------
#  Marker wrappers
# ---------------------------------------------------------------------------

def write_points(path: str | Path, points,
                 point_data: Optional[Mapping[str, object]] = None) -> Path:
    """Write a set of marker points (one ``VTK_VERTEX`` cell each)."""
    pts = _as_numpy(points, float).reshape(-1, 3)
    cells = np.arange(pts.shape[0], dtype=np.int64).reshape(-1, 1)
    return write_vtu(path, pts, cells, VTK_VERTEX, point_data=point_data)


def write_polyline(path: str | Path, points,
                   point_data: Optional[Mapping[str, object]] = None) -> Path:
    """Write an ordered list of points as a single polyline."""
    pts = _as_numpy(points, float).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise ValueError("a polyline needs at least two points")
    cells = np.arange(pts.shape[0], dtype=np.int64).reshape(1, -1)
    return write_vtu(path, pts, cells, VTK_POLY_LINE, point_data=point_data)


# ---------------------------------------------------------------------------
#  Scene and series
# ---------------------------------------------------------------------------

def write_scene(
    directory: str | Path,
    mesh,
    u=None,
    name: str = "solution",
    markers: Optional[Mapping[str, object]] = None,
    curves: Optional[Mapping[str, object]] = None,
    point_data: Optional[Mapping[str, object]] = None,
) -> Dict[str, Path]:
    """Write a whole ParaView scene into ``directory``.

    Produces ``<name>_reference.vtu``, and -- when ``u`` is given --
    ``<name>_deformed.vtu``, plus one file per entry of ``markers`` (point sets)
    and ``curves`` (polylines).  Open them together in ParaView to build a
    figure like figure 1 of Land et al. (2015).

    Parameters
    ----------
    markers, curves:
        ``name -> (n, 3)`` coordinate arrays, already in the configuration you
        want them drawn in.

    Returns
    -------
    ``name -> Path`` for every file written.
    """
    directory = Path(directory)
    written: Dict[str, Path] = {
        "reference": write_mesh(directory / f"{name}_reference.vtu", mesh)
    }
    if u is not None:
        written["deformed"] = write_deformed(directory / f"{name}_deformed.vtu",
                                             mesh, u, point_data=point_data)
    for label, pts in (markers or {}).items():
        written[label] = write_points(directory / f"{name}_{label}.vtu", pts)
    for label, pts in (curves or {}).items():
        written[label] = write_polyline(directory / f"{name}_{label}.vtu", pts)
    return written


def write_pvd(path: str | Path, files: Sequence[str | Path],
              times: Optional[Sequence[float]] = None) -> Path:
    """Write a ``.pvd`` collection so ParaView animates a series of ``.vtu`` files.

    Useful for load-step or time series; ``times`` defaults to ``0, 1, 2, ...``.
    Paths are stored relative to the ``.pvd`` file when possible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = list(times) if times is not None else list(range(len(files)))
    if len(times) != len(files):
        raise ValueError(f"got {len(files)} files but {len(times)} times")

    entries = []
    for t, f in zip(times, files):
        f = Path(f)
        try:
            f = f.relative_to(path.parent)
        except ValueError:
            pass
        entries.append(f'      <DataSet timestep="{t:.10g}" file="{f}"/>')

    path.write_text("\n".join([
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
        *entries,
        "  </Collection>",
        "</VTKFile>",
        "",
    ]))
    return path


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------

# The six quadrilateral faces of a VTK_HEXAHEDRON, in its node ordering.
_HEX_FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                       [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]])


def _surface(cells: np.ndarray) -> np.ndarray:
    """Outward skin of a hexahedral mesh: the faces belonging to one cell only."""
    faces = cells[:, _HEX_FACES].reshape(-1, 4)
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    return faces[counts[inverse.reshape(-1)] == 1]


def render_deformation(
    path: str | Path,
    mesh,
    u,
    points: Optional[Mapping[str, object]] = None,
    curves: Optional[Mapping[str, object]] = None,
    title: Optional[str] = None,
    elev: float = 18.0,
    azim: float = -72.0,
    alpha: float = 1.0,
    dpi: int = 110,
) -> Path:
    """Render the reference and deformed shapes to a single PNG.

    The undeformed body is drawn translucent grey and the deformed one coloured
    by displacement magnitude -- the arrangement used by figure 1 of Land et al.
    (2015).  Optional ``points`` and ``curves`` (``label -> (n, 3)`` arrays,
    already in the configuration you want them drawn in) are overlaid and
    labelled.  Lower ``alpha`` when the deformed body encloses the reference
    one, as an inflating chamber does, so both stay visible.

    Requires matplotlib, which is imported lazily so the rest of the module
    stays dependency-free.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X = _as_numpy(mesh.points, float).reshape(-1, 3)
    disp = _as_numpy(u, float).reshape(-1, 3)
    x = X + disp
    skin = _surface(_as_numpy(mesh.linear_cells(), np.int64))

    magnitude = np.linalg.norm(disp, axis=1)
    shading = magnitude[skin].mean(axis=1)
    colours = plt.get_cmap("viridis")(
        shading / shading.max() if shading.max() > 0 else shading)

    fig = plt.figure(figsize=(12.0, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(X[skin], facecolor="#c9c9c9",
                                         edgecolor="#8a8a8a", linewidths=0.1,
                                         alpha=0.55))
    ax.add_collection3d(Poly3DCollection(x[skin], facecolor=colours,
                                         edgecolor="k", linewidths=0.1,
                                         alpha=alpha))

    for label, pts in (curves or {}).items():
        ax.plot(*_as_numpy(pts, float).reshape(-1, 3).T, lw=2.0, label=label,
                zorder=10)
    for label, pts in (points or {}).items():
        ax.scatter(*_as_numpy(pts, float).reshape(-1, 3).T, s=45, label=label,
                   depthshade=False, zorder=11)

    # True proportions: a 10 x 1 x 1 beam should look like one, so the box
    # aspect follows the data extents rather than being forced to a cube.
    lo = np.minimum(X.min(axis=0), x.min(axis=0))
    hi = np.maximum(X.max(axis=0), x.max(axis=0))
    pad = 0.04 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))
    ax.view_init(elev=elev, azim=azim)
    # A slender body gets very few ticks on its short axes, or they collide.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_zlabel("z (mm)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    if points or curves:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
