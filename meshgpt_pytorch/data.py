from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, reduce

from beartype import beartype
from beartype.typing import Tuple, Union, Optional, Callable, Dict

from torchtyping import TensorType

# constants

Vertices = TensorType['nv', 3, float]   # 3 coordinates
Faces = TensorType['nf', 3, int]        # 3 vertices

# dataset

class DatasetFromTransforms(Dataset):
    @beartype
    def __init__(
        self,
        folder: str,
        transforms: Dict[str, Callable[Path, Tuple[Vertices, Faces]]]
    ):
        folder = Path(folder)
        assert folder.exists and folder.is_dir()
        self.folder = folder

        exts = transforms.keys()
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')
        assert len(self.paths) > 0

        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        ext = path.suffix[1:]
        fn = self.transforms[ext]

        return fn(path)

# tensor helper functions

def derive_face_edges_from_faces(
    faces: TensorType['b', 'nf', 3, int],
    pad_id = -1,
    neighbor_if_share_one_vertex = False,
    include_self = True
) -> TensorType['b', 'e', 2, int]:

    is_one_face, device = faces.ndim == 2, faces.device

    if is_one_face:
        faces = rearrange(faces, 'nf c -> 1 nf c')

    max_num_faces = faces.shape[1]
    face_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2

    all_edges = torch.stack(torch.meshgrid(
        torch.arange(max_num_faces, device = device),
        torch.arange(max_num_faces, device = device),
    indexing = 'ij'), dim = -1)

    face_masks = reduce(faces != pad_id, 'b nf c -> b nf', 'all')
    face_edges_masks = rearrange(face_masks, 'b i -> b i 1') & rearrange(face_masks, 'b j -> b 1 j')

    face_edges = []

    for face, face_edge_mask in zip(faces, face_edges_masks):

        shared_vertices = rearrange(face, 'i c -> i 1 c 1') == rearrange(face, 'j c -> 1 j 1 c')
        num_shared_vertices = shared_vertices.any(dim = -1).sum(dim = -1)
        is_neighbor_face = num_shared_vertices >= face_edges_vertices_threshold & face_edge_mask

        if not include_self:
            is_neighbor_face &= num_shared_vertices != 3

        face_edge = all_edges[is_neighbor_face]
        face_edges.append(face_edge)

    face_edges = pad_sequence(face_edges, padding_value = pad_id, batch_first = True)

    if is_one_face:
        face_edges = rearrange(face_edges, '1 e ij -> e ij')

    return face_edges

# custom collater

def custom_collate(data, pad_id = -1):
    is_dict = isinstance(data[0], dict)

    if is_dict:
        keys = data[0].keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        padded = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        output.append(padded)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output
