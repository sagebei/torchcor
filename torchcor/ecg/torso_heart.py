import numpy as np
from pathlib import Path
from torchcor.core import MeshReader, MeshWriter


class TorsoHeartMesh:
    def __init__(self):
        self.torso_nodes = None
        self.torso_elems = None
        self.torso_regions = None 
        self.torso_fibres = None

        self.heart_nodes = None
        self.heart_elems = None
        self.heart_regions = None 
        self.heart_fibres = None

        self.old_to_new = None
        self.heart_mesh_dir = None

    def extract_mesh_on_tags(self, nodes, elems, regions, fibres, tags=[24, 25, 34, 35, 36]):
        tags = np.array(list(tags), dtype=np.int64)
        mask = np.isin(regions, tags)

        elems_keep = elems[mask]
        fibres_keep = fibres[mask]
        regions_keep = regions[mask]

        old_nodes_index = np.unique(elems_keep.reshape(-1))
        heart_to_torso_node = np.sort(old_nodes_index)
        
        nodes_extracted = nodes[heart_to_torso_node]

        self.old_to_new = np.full((nodes.shape[0],), -1, dtype=np.int64)
        self.old_to_new[heart_to_torso_node] = np.arange(heart_to_torso_node.size, dtype=np.int64)
        elems_extracted = self.old_to_new[elems_keep]
        
        return nodes_extracted, elems_extracted, regions_keep, fibres_keep

    def load_torsor_mesh(self, torsor_mesh_dir="/data/Bei/Torso/HC2/mesh", unit_conversion=1000):
        reader = MeshReader(torsor_mesh_dir)
        self.torso_nodes, self.torso_elems, self.torso_regions, self.torso_fibres = reader.read(unit_conversion=unit_conversion)
        return self.torso_nodes, self.torso_elems, self.torso_regions, self.torso_fibres
    
    def load_heart_mesh(self, heart_mesh_dir="/data/Bei/Torso/HC2/heart", unit_conversion=1000):
        reader = MeshReader(heart_mesh_dir)
        self.heart_nodes, self.heart_elems, self.heart_regions, self.heart_fibres = reader.read(unit_conversion=unit_conversion)
        return self.heart_nodes, self.heart_elems, self.heart_regions, self.heart_fibres
    
    def extract_heart_mesh(self, heart_mesh_dir="/data/Bei/Torso/HC2/heart", filename="1", tags=[24, 34, 36]):
        self.heart_mesh_dir = Path(heart_mesh_dir)
        self.heart_nodes, self.heart_elems, self.heart_regions, self.heart_fibres = self.extract_mesh_on_tags(self.torso_nodes, self.torso_elems, self.torso_regions, self.torso_fibres, tags)
        writer = MeshWriter(mesh_dir=self.heart_mesh_dir, filename=filename)

        writer.write(self.heart_nodes, self.heart_elems, self.heart_regions, self.heart_fibres)

    def load_stimulus_region(self, vtx_filepath):
        with Path(vtx_filepath).open("r") as f:
            region_size = int(f.readline().strip())
        region = np.loadtxt(vtx_filepath, dtype=int, skiprows=2)

        if len(region) != region_size:
            raise Exception(f"Error loading {vtx_filepath}")
        
        return region
    
    def save_stimulus_region(self, region, vtx_filepath):
        region = region.astype(np.int64).reshape(-1)
        with Path(vtx_filepath).open("w") as f:
            f.write(f"{len(region)}\n")
            f.write("extra\n")
            np.savetxt(f, region, fmt="%d")

    def convert_pacing_sites(self, pacing_sites_dir="/data/Bei/Torso/HC2/HPS"):
        pacing_sites_dir = Path(pacing_sites_dir)
        pacing_folder = self.heart_mesh_dir / "pacing" 
        pacing_folder.mkdir(exist_ok=True, parents=True)

        for filepath in pacing_sites_dir.iterdir():
            if filepath.suffix == ".vtx":
                region = self.load_stimulus_region(filepath)
                new_region = self.old_to_new[region]
                self.save_stimulus_region(new_region, pacing_folder / filepath.name)


if __name__ == "__main__":
    torsor_mesh_dir = "/data/Bei/Torso/HC2/mesh"
    heart_mesh_dir = "/data/Bei/Torso/HC2/heart"
    
    thm = TorsoHeartMesh()
    thm.load_torsor_mesh(torsor_mesh_dir=torsor_mesh_dir, unit_conversion=1000)
    thm.extract_heart_mesh(heart_mesh_dir=heart_mesh_dir, filename="1", tags=[24, 25, 34, 35, 36])
    thm.convert_pacing_sites(pacing_sites_dir="/data/Bei/Torso/HC2/HPS")