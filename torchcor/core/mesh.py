from pathlib import Path
import numpy as np
import torch

def fmt(v, decimals=10):
    s = f"{float(v):.{decimals}f}"
    return s.rstrip("0").rstrip(".")


class Elems:
    def __init__(self):
        self.Ln = None
        self.Ln_idx = None

        self.Tr = None
        self.Tr_idx = None

        self.Tt = None
        self.Tt_idx = None

    def to_torch(self, device):
        if self.Ln is not None:
            self.Ln = torch.from_numpy(self.Ln).to(dtype=torch.long, device=device)
        if self.Ln_idx is not None:
            self.Ln_idx = torch.from_numpy(self.Ln_idx).to(dtype=torch.long, device=device)

        if self.Tr is not None:
            self.Tr = torch.from_numpy(self.Tr).to(dtype=torch.long, device=device)
        if self.Tr_idx is not None:
            self.Tr_idx = torch.from_numpy(self.Tr_idx).to(dtype=torch.long, device=device)

        if self.Tt is not None:
            self.Tt = torch.from_numpy(self.Tt).to(dtype=torch.long, device=device)
        if self.Tt_idx is not None:
            self.Tt_idx = torch.from_numpy(self.Tt_idx).to(dtype=torch.long, device=device)

        return self

class MeshReader:
    def __init__(self, mesh_dir: str):
        self.mesh_dir = Path(mesh_dir)

        self.node_file: Path = None
        self.elem_file: Path = None
        self.fibre_file: Path = None
        
        for file_path in self.mesh_dir.iterdir():
            if file_path.suffix == ".pts":
                self.node_file = file_path
            elif file_path.suffix == ".elem":
                self.elem_file = file_path
            elif file_path.suffix == ".lon":
                self.fibre_file = file_path

        self.nodes: np.array = None
        self.elems = Elems()
        self.elems_index = Elems()
        self.regions: np.array = None
        self.fibres: np.array = None

    def read_nodes(self, unit_conversion=1000):
        with self.node_file.open("r") as f:
            num_pts_expected = int(f.readline().strip()) 
        
        nodes = np.loadtxt(self.node_file, dtype=float, skiprows=1)
        num_pts_actual = nodes.shape[0]
        if num_pts_actual != num_pts_expected:
            raise ValueError(f"Mismatch in number of nodes: expected {num_pts_expected}, but found {num_pts_actual}")

        self.nodes = nodes / unit_conversion

    def read_elems(self):
        with self.elem_file.open("r") as f:
            num_elems_expected = int(f.readline().strip())
            lines = f.readlines()

        num_elems_actual = len(lines)
        if num_elems_actual != num_elems_expected:
            raise ValueError(f"Mismatch: expected {num_elems_expected}, but found {num_elems_actual}")

        elem_type_map = {"Ln": 2, "Tr": 3, "Tt": 4}

        elem_types = []
        elem_nodes = []
        regions = []

        for i, line in enumerate(lines):
            parts = line.strip().split()
            etype = parts[0]
            if etype not in elem_type_map:
                raise ValueError(f"Unknown element type '{etype}' at line {i + 2}")

            n_nodes = elem_type_map[etype]
            node_indices = [int(x) for x in parts[1 : 1 + n_nodes]]
            region = int(parts[1 + n_nodes])

            elem_types.append(etype)
            elem_nodes.append(node_indices)
            regions.append(region)

        self.elem_types = np.array(elem_types, dtype="U2")
        self.regions = np.array(regions, dtype=int)
        
        for etype in elem_type_map:
            mask = self.elem_types == etype
            if mask.any():
                indices = np.where(mask)[0]
                if etype == "Ln":
                    self.elems.Ln = np.array([elem_nodes[i] for i in indices], dtype=int)
                    self.elems.Ln_idx = indices
                elif etype == "Tr":
                    self.elems.Tr = np.array([elem_nodes[i] for i in indices], dtype=int)
                    self.elems.Tr_idx = indices
                elif etype == "Tt":
                    self.elems.Tt = np.array([elem_nodes[i] for i in indices], dtype=int)
                    self.elems.Tt_idx = indices


    def read_fibres(self):
        fibres = np.loadtxt(self.fibre_file, dtype=np.float32, skiprows=1)
        norms = np.linalg.norm(fibres, axis=1, keepdims=True)
        
        mask = norms[:, 0] > 1e-10
        fibres[mask] /= norms[mask]

        self.fibres = fibres
    
    def read(self, unit_conversion=1000):
        self.read_nodes(unit_conversion)
        self.read_elems()
        self.read_fibres()

        return self.nodes, self.elems, self.regions, self.fibres

    def compute_edges(self):
        pass



class MeshWriter:
    def __init__(self, mesh_dir, filename):
        self.mesh_dir = Path(mesh_dir)
        self.filename = filename

    def write(self, nodes, elems, regions, fibres=None, unit_conversion=1000):
        """
        nodes:   (N, 3) float
        elems:   (M, 4) int for volume tets OR (M, 3) int for surface tris
        regions: (M,)   int
        fibres:  (M, 3) float (optional, typically for volume meshes)
        """
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        nodes = np.asarray(nodes, dtype=np.float64)
        elems = np.asarray(elems, dtype=np.int64)
        regions = np.asarray(regions, dtype=np.int64).reshape(-1)

        # ---- .pts ----
        pts_file = self.mesh_dir / f"{self.filename}.pts"
        nodes_out = (nodes * unit_conversion).astype(np.float32)

        with pts_file.open("w") as f:
            f.write(f"{nodes_out.shape[0]}\n")
            for x, y, z in nodes_out:
                f.write(f"{fmt(x)} {fmt(y)} {fmt(z)}\n")

        # ---- .elem ----
        elem_file = self.mesh_dir / f"{self.filename}.elem"
        etype = "Tt" if elems.shape[1] == 4 else "Tr"

        with elem_file.open("w") as f:
            f.write(f"{elems.shape[0]}\n")
            if elems.shape[1] == 4:
                for (n0, n1, n2, n3), r in zip(elems, regions):
                    f.write(f"{etype} {n0} {n1} {n2} {n3} {r}\n")
            else:
                for (n0, n1, n2), r in zip(elems, regions):
                    f.write(f"{etype} {n0} {n1} {n2} {r}\n")

        # ---- .lon  ----
        if fibres is not None:
            lon_file = self.mesh_dir / f"{self.filename}.lon"
            fibres_out = np.asarray(fibres, dtype=np.float32)

            with lon_file.open("w") as f:
                f.write(f"{fibres_out.shape[0]}\n")
                for fx, fy, fz in fibres_out:
                    f.write(f"{fmt(fx)} {fmt(fy)} {fmt(fz)}\n")


if __name__ == "__main__":
    import time
    start_time = time.time()
    reader = MeshReader("/home/bzhou6/Data/atrium/Case_1")
    # reader = MeshReader("/home/bzhou6/Data/ventricle/")
    reader.read_nodes()
    reader.read_elems()
    reader.read_fibres()
    print(time.time() - start_time)

    print(reader.nodes[0], reader.elems.Tr[0], reader.regions[0], reader.fibres[0])