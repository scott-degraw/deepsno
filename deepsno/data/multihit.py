import glob
from typing import Callable, Iterable
from warnings import warn

import awkward as ak
import numba as nb
import numpy as np
import torch
import uproot
from torch.utils import _pytree as pytree
from torch.utils.data import IterableDataset


class UprootMultiFileDataset(IterableDataset):
    def __init__(
        self,
        file_paths: str | Iterable[str],
        tree_name: str,
        expressions: Iterable[str] | None = None,
        cut: str | None = None,
        seed: int = 42,
        buffer_size: int = 100,
        debug: bool = False,
    ) -> None:
        if expressions is None:
            expressions = []
        self.file_paths = file_paths
        if isinstance(file_paths, str):
            file_paths = glob.glob(file_paths)
        self.tree_name = tree_name
        self.expressions = expressions
        self.cut = cut
        self.seed = seed
        self.buffer_size = buffer_size
        self.generator = None
        self.debug = debug

        self.length = 0

        with uproot.open({next(iter(file_paths)): self.tree_name}, cut=self.cut) as ntuple:
            self.length += len(ntuple["npe"].array())

        self.length *= len(file_paths)
        if self.debug:
            self.debug_print("Initializing")

        self.events_yielded = 0

    def __len__(self) -> int:
        return self.length

    def debug_print(self, msg: str) -> None:
        if self.debug:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                n_workers = 1
            else:
                worker_id = worker_info.id
                n_workers = worker_info.num_workers
            print(f"Worker {worker_id + 1} of {n_workers}: {msg}")

    def __iter__(self):
        file_paths = glob.glob(self.file_paths) if isinstance(self.file_paths, str) else self.file_paths
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            n_workers = 1
        else:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers

        if self.debug:
            print(f"Worker {worker_id + 1} of {n_workers} starting.")

        self.seed += worker_id

        if self.generator is None:
            self.generator = np.random.default_rng(self.seed)

        file_slice = slice(
            len(file_paths) * worker_id // n_workers,
            len(file_paths) * (worker_id + 1) // n_workers,
        )

        file_paths = file_paths[file_slice]
        if len(file_paths) == 0:
            raise ValueError("No files assigned to this worker!")

        shuffled_file_indices = self.generator.permutation(len(file_paths))

        buffer = []
        self.n_entries = 0

        for file_index in shuffled_file_indices:
            file = file_paths[file_index]

            with uproot.open(file) as ntuple:
                arrays = ntuple[self.tree_name].arrays(self.expressions)

            for entry in arrays:
                entry = (entry, file)
                if len(buffer) < self.buffer_size:
                    buffer.append(entry)
                    continue
                elif len(buffer) > self.buffer_size:
                    raise ValueError(f"Buffer size exceeded! Size is {len(buffer)} but should be {self.buffer_size}.")

                buffer_i = self.generator.choice(self.buffer_size)
                if len(buffer) < 1:
                    raise ValueError("Buffer is empty!")
                if len(buffer) != self.buffer_size:
                    raise ValueError(f"Buffer is not full!. Size is {len(buffer)} but should be {self.buffer_size}.")

                self.debug_print(f"Yielding event {self.n_entries}")

                self.n_entries += 1
                yield buffer[buffer_i]

                buffer[buffer_i] = entry

        self.debug_print("Flushing buffer")
        # Flush out the rest of the buffer
        permutations = self.generator.permutation(self.buffer_size)
        for buffer_i in permutations:
            yield buffer[buffer_i]


def pad_array(
    array: np.ndarray, pad_length: int, axis: int | None = None, generator: np.random.Generator | None = None
) -> np.ndarray:
    if axis is None:
        array = array.ravel()
        axis = 0
    if generator is None:
        generator = np.random.default_rng()
    pad_width = pad_length - array.shape[axis]

    if pad_width == 0:
        return array
    if pad_width < 0:
        sampled_indices = generator.choice(array.shape[axis], size=pad_length, replace=False)
        return np.take(array, sampled_indices, axis=axis)
    else:
        indexer = array.ndim * [slice(None)]
        indexer[axis] = slice(None, array.shape[axis])
        padded_shape = list(array.shape)
        padded_shape[axis] = pad_length
        padded_array = np.zeros_like(array, shape=padded_shape)
        padded_array[tuple(indexer)] = array

        return padded_array


@nb.njit
def voxelise_line(
    first_pos: np.ndarray,
    last_pos: np.ndarray,
    edges: list[np.ndarray],
    active: np.ndarray | None = None,
):
    if active is None:
        active = np.zeros((len(edges[0]), len(edges[1]), len(edges[2]), len(edges[3])), dtype=np.bool)

    voxel_centers = [0.5 * (edges[:-1] + edges[1:]) for edges in edges]

    # For each coordinate planes we will find the intersections
    for coord_i in range(len(edges)):
        # The "z" refers to the current coordinate axis being processed
        # Find the endpoints on coord_i
        z_min = min(first_pos[coord_i], last_pos[coord_i])
        z_max = max(first_pos[coord_i], last_pos[coord_i])

        # Find the low and high voxel plane intersections
        low_z_i = np.digitize(z_min, edges[coord_i]) - 1
        high_z_i = np.digitize(z_max, edges[coord_i]) - 1
        # Clip to valid range
        low_z_i = max(low_z_i, 0)
        high_z_i = min(high_z_i, len(edges[coord_i]) - 1)

        # Find the plane intersections
        z_edge_i = np.arange(low_z_i, high_z_i + 1)
        z_planes = edges[coord_i][low_z_i : high_z_i + 1]

        # Parametrise straight line with lambd in [0, 1] and find intersections with planes
        lambd = (z_planes - first_pos[coord_i]) / (last_pos[coord_i] - first_pos[coord_i])
        lambd = np.clip(lambd, 0, 1)

        # Find the intercepts in all len(edges) coordinates
        # (len(edges), N_intercepts)
        intercepts = lambd * (last_pos[:, None] - first_pos[:, None]) + first_pos[:, None]

        # # Find the voxel indices for each intercept
        # # (len(edges), N_intercepts)
        edge_is = np.empty_like(intercepts, dtype=np.int64)
        # for i in set(range(len(edges))) - {coord_i}:
        for i in range(len(edges)):
            if i == coord_i:
                continue
            edge_is[i] = np.digitize(intercepts[i], edges[i]) - 1
        # Due to numerical precision issues, sometimes the coord_i indices are off by 1
        # Use the fact that we know what they should be to correct this
        edge_is[coord_i] = z_edge_i

        # Only include intercepts that are within the voxel grid in all len(edges) coordinates
        # (N_intercepts,)
        edge_valid = np.ones_like(edge_is[0], np.bool)
        for i in range(len(edges)):
            edge_valid &= (0 <= edge_is[i]) & (edge_is[i] < len(voxel_centers[i]))

        # Get the voxel centers for each valid intercept and add to list
        for j in range(edge_is.shape[1]):
            if edge_valid[j]:
                active[edge_is[0, j], edge_is[1, j], edge_is[2, j], edge_is[3, j]] = True

        edge_is[coord_i] -= 1
        for j in range(1, edge_is.shape[1]):
            if edge_valid[j]:
                active[edge_is[0, j], edge_is[1, j], edge_is[2, j], edge_is[3, j]] = True

    return active


@nb.njit()
def voxelise_track(track_positions: np.ndarray, edges: list[np.ndarray], active: np.ndarray | None = None):
    for i in range(track_positions.shape[0] - 1):
        first_pos = track_positions[i]
        last_pos = track_positions[i + 1]

        active = voxelise_line(
            first_pos=first_pos,
            last_pos=last_pos,
            edges=edges,
            active=active,
        )

    return active


def voxelise_tracks(tracks: ak.Array, edges: list[np.ndarray], active: np.ndarray | None = None):
    for track in tracks:
        positions = track["steps"]["position"].to_numpy()
        times = track["steps"]["time"].to_numpy()

        positions = np.concat([positions, times[:, None]], axis=1)
        active = voxelise_track(positions, edges, active)

    return active


class MultiHitDataset(UprootMultiFileDataset):
    def __init__(
        self,
        waveform_generator: Callable,
        max_context_len: int,
        max_n_vertices: int,
        min_deposited_energy: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.waveform_generator = waveform_generator

        self.expressions += "hit_times", "npe", "mc_index", "tracks*"
        self.max_context_len = max_context_len
        self.max_n_vertices = max_n_vertices
        self.min_deposited_energy = min_deposited_energy

        radius = 6000
        self.edges = [np.arange(-radius, radius, step=100, dtype=np.float32) for _ in range(3)]
        self.edges += [np.arange(0, 40, step=0.5, dtype=np.float32)]
        self.centers = [0.5 * (e[:-1] + e[1:]) for e in self.edges]
        voxel_dims = [len(e) - 1 for e in self.edges]

        self.active_voxels = np.zeros(voxel_dims, dtype=bool)

    def __iter__(self):
        for entry, file_path in super().__iter__():
            hits_per_pmt = ak.num(entry["hit_times"]).to_numpy()
            hit_pmt_ids = np.nonzero(hits_per_pmt)[0]
            if len(hit_pmt_ids) == 0:
                warn(f"No hits found in event {entry['mc_index'].item()} in file {file_path}")

            waveforms = self.waveform_generator(entry["hit_times"])[hit_pmt_ids]

            inputs = {"pmt_ids": hit_pmt_ids, "waveforms": waveforms}

            if np.sum(inputs["waveforms"]) == 0:
                warn(f"No waveform data in event {entry['mc_index'].item()} in file {file_path}")

            inputs = pytree.tree_map(
                lambda x: pad_array(x, pad_length=self.max_context_len, axis=0, generator=self.generator), inputs
            )

            tracks = entry["tracks"][entry["tracks"]["deposited_energy"] >= self.min_deposited_energy]
            if len(tracks) == 0:
                warn(
                    f"No tracks with deposited energy > {self.min_deposited_energy} in event {entry['mc_index'].item()} in file {file_path}"
                )

            self.active_voxels.fill(False)
            self.active_voxels = voxelise_tracks(tracks, self.edges, self.active_voxels)
            vertex_indices = np.nonzero(self.active_voxels)
            vertex_positions = np.stack([self.centers[i][vertex_indices[i]] for i in range(len(self.edges))], axis=1)
            if vertex_positions.shape[0] == 0:
                warn(f"No vertices found in event {entry['mc_index'].item()} in file {file_path}")

            exists = np.ones(vertex_positions.shape[0], dtype=bool)

            exists = pad_array(exists, pad_length=self.max_n_vertices, axis=0, generator=self.generator)
            vertices = pad_array(vertex_positions, pad_length=self.max_n_vertices, axis=0, generator=self.generator)

            vertices = {"position": vertices[:, :3], "time": vertices[:, 3], "exists": exists}

            # Shuffle around the vertices so they're not in any particular order
            vertex_shuffle_i = self.generator.permutation(self.max_n_vertices)
            vertices = pytree.tree_map(lambda x: x[vertex_shuffle_i], vertices)
            vertices = pytree.tree_map(torch.from_numpy, vertices)

            truth = {
                **vertices,
                "mc_index": entry["mc_index"].item(),
                "npe": entry["npe"].item(),
                "file_path": file_path,
            }

            inputs = pytree.tree_map(torch.from_numpy, inputs)

            yield inputs, truth
