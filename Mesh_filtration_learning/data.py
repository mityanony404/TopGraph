import torch
import numpy as np
import torch_geometric
import torch_geometric.data
from torch_geometric import transforms
from torch_geometric import utils
import trimesh
import os
from collections import Counter
from tqdm import tqdm


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        assert isinstance(x, list)
        self.data = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    def __iter__(self):
        for i in range(len(self.data)):
            yield self.data[i]


class MyOwnDataset(torch_geometric.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, phase='train', verbose=True):
        self.phase = phase
        self.targets = []
        self.avg_num_nodes = None
        self.avg_num_edges = None
        self.num_classes = None
        self.verbose = verbose
        self.meshes = []
        super().__init__(root, transform, pre_transform)

    # @property
    # def raw_file_names(self):
    #     classes, class_to_id = find_classes(self.root)
    #     meshes = make_dataset_by_class(self.root, class_to_id, phase=self.phase)
    #     self.targets = [m[1] for m in meshes]
    #     return [m[0] for m in meshes]

    @property
    def processed_file_names(self):
        classes, class_to_id = find_classes(self.root)
        self.meshes = make_dataset_by_class(self.root, class_to_id, phase=self.phase)
        return [f'{self.phase}_data_{i + 1}.pt' for i in range(len(self.meshes))]

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)
    #     ...
    @property
    def processed_dir(self) -> str:
        processed_dir = os.path.join("processed_datasets", self.root)
        return processed_dir

    def process(self):
        idx = 0
        num_nodes = []
        num_edges = []
        num_faces = []
        face2edge = transforms.FaceToEdge(remove_faces=False)

        for m, y in self.meshes:
            # Read data from `raw_path`.
            # print(f'{m}')
            self.targets.append(y)
            mesh = trimesh.load(m)
            vertex_normal = mesh.vertex_normals.copy()
            mesh_data = utils.from_trimesh(mesh)
            mesh_data = face2edge(mesh_data)
            mesh_data.y = torch.tensor(y, dtype=torch.long)
            mesh_data.pos = torch.cat([mesh_data.pos, torch.tensor(vertex_normal, dtype=torch.float)], dim=1)
            mesh_data.num_nodes = mesh_data.pos.size(0)
            boundary_edges = get_boundary_info(mesh_data)
            mesh_data.boundary_edges = boundary_edges
            num_nodes.append(mesh_data.num_nodes)
            num_edges.append(boundary_edges.size(0))
            num_faces.append(mesh_data.face.size(1))

            if self.pre_transform is not None:
                mesh_data = self.pre_transform(mesh_data)

            torch.save(mesh_data, os.path.join(self.processed_dir, f'{self.phase}_data_{idx}.pt'))
            idx += 1

        avg_num_nodes = np.mean(num_nodes)
        avg_num_edges = np.mean(num_edges)
        avg_num_faces = np.mean(num_faces)
        self.num_classes = len(set(self.targets))

        if self.verbose:
            print("# Dataset: ", self.root)
            print('# num samples: ', len(self.meshes))
            print('# num classes: ', self.num_classes)
            # print('#')
            # print('# max node degree: ', self.max_node_deg)
            # print('# num node labels: ', self.num_node_lab)
            print('#')
            print('# avg number of nodes: ', avg_num_nodes)
            print('# avg number of edges: ', avg_num_edges)
            print('# avg number of faces: ', avg_num_faces)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.phase}_data_{idx}.pt'))
        return data


def find_classes(dir_name):
    classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset_by_class(dir_name, class_to_idx, phase):
    meshes = []
    dir_name = os.path.expanduser(dir_name)
    for target in sorted(os.listdir(dir_name)):
        d = os.path.join(dir_name, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if root.count(phase) == 1:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    meshes.append(item)
    return meshes


def get_boundary_info(m):
    e = m.edge_index.permute(1, 0)
    e = e.sort(1)
    e = e[0].tolist()
    e = set([tuple(ee) for ee in e])
    return torch.tensor([ee for ee in e], dtype=torch.long)


def dataset_factory(dir_name, verbose=True, phase='train'):
    x = []
    classes, class_to_id = find_classes(dir_name)
    meshes = make_dataset_by_class(dir_name, class_to_id, phase=phase)
    face2edge = transforms.FaceToEdge(remove_faces=True)
    for m, y in meshes:
        # print(f'{m}')
        mesh = trimesh.load(m)
        vertex_normal = mesh.vertex_normals.copy()
        mesh_data = utils.from_trimesh(mesh)
        mesh_data = face2edge(mesh_data)
        mesh_data.y = torch.tensor(y, dtype=torch.long)
        mesh_data.pos = torch.cat([mesh_data.pos, torch.tensor(vertex_normal, dtype=torch.float)], dim=1)
        mesh_data.num_nodes = mesh_data.pos.size(0)
        x.append(mesh_data)
    # dataset = SimpleDataset(x)
    # dataset.num_classes = len(set(classes))
    dataset = enhance_mesh_dataset(x)
    ds_name = dir_name
    if verbose:
        print("# Dataset: ", ds_name)
        print('# num samples: ', len(dataset))
        print('# num classes: ', dataset.num_classes)
        print('#')
        print('# max node degree: ', dataset.max_node_deg)
        print('# num node labels: ', dataset.num_node_lab)
        print('#')
        print('# avg number of nodes: ', dataset.avg_num_nodes)
        print('# avg number of edges: ', dataset.avg_num_edges)
    return dataset


def enhance_mesh_dataset(ds):
    x = []
    targets = []
    max_degree_by_graph = []
    num_nodes = []
    num_edges = []

    for mesh_data in ds:
        targets.append(mesh_data.y.item())
        boundary_edges = get_boundary_info(mesh_data)
        mesh_data.boundary_edges = boundary_edges
        num_nodes.append(mesh_data.num_nodes)
        num_edges.append(boundary_edges.size(0))
        x.append(mesh_data)

    new_ds = SimpleDataset(x)
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes = len(set(targets))
    return new_ds
