import glob
from typing import Iterable
from warnings import warn

import awkward as ak
import fsspec
import numba as nb
import numpy as np
import torch
import torch.distributed as dist
import uproot
from torch.utils import _pytree as pytree
from torch.utils.data import IterableDataset


class UprootMultiFileDataset(IterableDataset):
    def __init__(
        self,
        file_paths: str | Iterable[str],
        tree_name: str,
        expressions: Iterable[str] | None = None,
        filter_name: Iterable[str] | None = None,
        cut: str | None = None,
        seed: int = 42,
        buffer_size: int = 100,
        shuffle: bool = True,
        cache: bool | str = False,
        debug: bool = False,
    ) -> None:
        if expressions is None:
            expressions = set()
        if isinstance(file_paths, str):
            self.file_paths = glob.glob(file_paths)
            if len(self.file_paths) == 0:
                raise FileNotFoundError(f"No files found in: {file_paths}")
        else:
            self.file_paths = file_paths
        self.tree_name = tree_name
        self.expressions = set(expressions) if expressions is not None else set()
        self.filter_name = set(filter_name) if filter_name is not None else set()
        self.cut = cut
        self.seed = seed
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.generator = None
        self.debug = debug
        self.cache = cache

        if self.debug:
            self.debug_print("Initializing")

        self.events_yielded = 0

        self.length = None

    def __len__(self) -> int:
        if self.length is None:
            for path in self.file_paths:
                with uproot.open(path)[self.tree_name] as tree:
                    self.length += tree.num_entries
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
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No files found in: {self.file_paths}")
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            n_workers = 1
        else:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers

        # DDP rank-based sharding: each rank gets a disjoint subset of files.
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Combine rank and local-worker dimensions into a single global index.
        global_worker_id = rank * n_workers + worker_id
        total_workers = world_size * n_workers

        if self.debug:
            print(f"Worker {worker_id + 1} of {n_workers} (rank {rank}/{world_size}) starting.")

        if self.generator is None:
            epoch_seed = torch.initial_seed() % (2**31)
            self.generator = np.random.default_rng(self.seed + global_worker_id + epoch_seed)

        file_slice = slice(
            len(file_paths) * global_worker_id // total_workers,
            len(file_paths) * (global_worker_id + 1) // total_workers,
        )

        file_paths = file_paths[file_slice]
        if len(file_paths) == 0:
            raise ValueError("No files assigned to this worker!")

        file_indices = self.generator.permutation(len(file_paths)) if self.shuffle else range(len(file_paths))

        buffer = []
        self.n_entries = 0

        if self.cache:
            cache_storage = self.cache if isinstance(self.cache, str) else "data_cache"
            fs = fsspec.filesystem("simplecache", target_protocol="file", cache_storage=cache_storage)
            open_context = fs.open
        else:
            open_context = open

        for file_index in file_indices:
            file = file_paths[file_index]

            with open_context(file, mode="rb") as f:
                with uproot.open(f) as ntuple:
                    expr = self.expressions if self.expressions else None
                    filter_name = self.filter_name if self.filter_name else None
                    arrays = ntuple[self.tree_name].arrays(expr, filter_name=filter_name)

            empty = True
            for entry in arrays:
                empty = False
                entry = (entry, file)
                if not self.shuffle:
                    yield entry
                    continue
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

            if empty:
                raise ValueError(f"No entries found in file {file}!")

        if self.shuffle:
            self.debug_print("Flushing buffer")
            permutations = self.generator.permutation(len(buffer))
            for buffer_i in permutations:
                yield buffer[buffer_i]


def pad_array(
    array: np.ndarray,
    pad_length: int,
    axis: int | None = None,
    generator: np.random.Generator | None = None,
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
def hist_jagged(x: ak.Array, bin_width: float, low: float, counts: np.ndarray) -> np.ndarray:
    for i in range(len(x)):
        row = x[i]
        for j in range(len(row)):
            bin_i = int((row[j] - low) / bin_width)
            if (0 <= bin_i) and (bin_i < counts.shape[1]):
                counts[i, bin_i] += 1

    return counts


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
def voxelise_track(
    track_positions: np.ndarray,
    edges: list[np.ndarray],
    active: np.ndarray | None = None,
):
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


def voxel_vertices(active: np.ndarray, centers: list[np.ndarray]):
    vertex_indices = np.nonzero(active)
    vertex_positions = np.stack([centers[i][vertex_indices[i]] for i in range(len(centers))], axis=1)
    return vertex_positions


def voxelise_points(points: np.ndarray, edges: list[np.ndarray], *aux_values) -> np.ndarray:
    spacings = np.array([e[1] - e[0] for e in edges], dtype=np.float32)
    lows = np.array([e[0] for e in edges], dtype=np.float32)
    max_edge_indices = np.array([len(e) - 1 for e in edges], dtype=np.int64)
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]

    indices = np.floor((points - lows) / spacings)
    good_indices = (0 <= indices) & (indices < max_edge_indices - 1)
    good_indices = np.all(good_indices, axis=1)
    indices = indices[good_indices].astype(np.int64)
    indices, uniq_2_non_uniq_indices = np.unique(indices, axis=0, return_inverse=True)

    vertex_positions = np.stack([centers[i][indices[:, i]] for i in range(len(edges))], axis=1)

    if aux_values:
        reduced_aux_values = []
        for value in aux_values:
            value = value[good_indices]

            sort_i = np.argsort(uniq_2_non_uniq_indices)
            uniq_2_non_uniq_indices = uniq_2_non_uniq_indices[sort_i]
            value = value[sort_i]

            # For each label on non unique vector find the number of these elements and their starting point
            _, uniq_label_start_indices = np.unique(uniq_2_non_uniq_indices, return_index=True)
            value = np.add.reduceat(value, uniq_label_start_indices)

            reduced_aux_values.append(value)

        return vertex_positions, *reduced_aux_values
    return vertex_positions


class MultiHitDatasetBase(UprootMultiFileDataset):
    """
    Shared base for MultiHit datasets.

    Handles all data loading, hit filtering, log-time quantisation,
    truncation/padding, and truth-building.  Subclasses implement
    :meth:`_make_pmt_inputs` to choose how PMT data are represented.
    """

    def __init__(
        self,
        waveform_range: tuple[float, float],
        n_waveform_bins: int,
        n_pmts: int,
        radius: float,
        pos_spacing: float,
        time_spacing: float,
        max_context_len: int,
        max_n_vertices: int,
        min_energy: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.waveform_range = waveform_range
        self.waveform_dt = (waveform_range[1] - waveform_range[0]) / n_waveform_bins
        self.n_waveform_bins = n_waveform_bins
        self.n_pmts = n_pmts

        self.filter_name.update(["hit_times", "npe", "mc_index", "tracks*"])
        self.max_context_len = max_context_len
        self.max_n_vertices = max_n_vertices
        self.min_energy = min_energy

        self.pos_spacing = pos_spacing
        self.time_spacing = time_spacing
        self.grid_spacing = np.array(
            [self.pos_spacing, self.pos_spacing, self.pos_spacing, self.time_spacing],
            dtype=np.float32,
        )
        self.edges = [
            np.arange(-radius, radius + self.pos_spacing, self.pos_spacing, dtype=np.float32) for _ in range(3)
        ]
        self.edges += [np.arange(0, 40 + self.time_spacing, self.time_spacing, dtype=np.float32)]
        self.lows = np.array([e[0] for e in self.edges], dtype=np.float32)
        self.centers = [0.5 * (e[:-1] + e[1:]) for e in self.edges]
        self.max_edge_indices = np.array([len(e) - 1 for e in self.edges], dtype=np.int64)

    def _make_pmt_inputs(self, pmt_ids: np.ndarray, hit_times: np.ndarray) -> dict[str, np.ndarray]:
        """
        Build the ``inputs`` dict from sorted, padded ``pmt_ids`` and
        ``hit_times`` arrays (each of length ``max_context_len``).

        Subclasses must override this method.
        """
        raise NotImplementedError

    def __iter__(self):
        self.time_edges = np.linspace(2, 6, self.n_waveform_bins + 1)

        for entry, file_path in super().__iter__():
            # --- hit extraction & filtering ---
            hits_per_pmt = ak.num(entry["hit_times"]).to_numpy()
            pmt_ids = np.arange(len(entry["hit_times"]))
            pmt_ids = np.repeat(pmt_ids, hits_per_pmt)
            hit_times = ak.flatten(entry["hit_times"]).to_numpy()

            selector = (hit_times > 0) & (hit_times < 300)
            pmt_ids = pmt_ids[selector]
            hit_times = hit_times[selector]

            if np.any(~np.isfinite(np.log(hit_times))):
                raise ValueError("Bad log hit times")

            # --- truncate to max_context_len ---
            if len(pmt_ids) > self.max_context_len:
                shuffle_indices = self.generator.choice(len(pmt_ids), size=self.max_context_len, replace=False)
                pmt_ids = pmt_ids[shuffle_indices]
                hit_times = hit_times[shuffle_indices]

            # sort by PMT id so subclasses can rely on ordering
            sort_idx = np.argsort(pmt_ids)
            pmt_ids = pmt_ids[sort_idx]
            hit_times = hit_times[sort_idx]

            # --- subclass-specific PMT representation ---
            inputs = self._make_pmt_inputs(pmt_ids, hit_times)

            # --- truth building ---
            tracks = entry["tracks"][entry["tracks"]["deposited_energy"] > self.min_energy]
            if len(tracks) == 0:
                warn(
                    f"No tracks with deposited energy > {self.min_energy} in event "
                    f"{entry['mc_index'].item()} in file {file_path}"
                )

            vertex_positions = np.concatenate(
                [
                    ak.flatten(tracks["steps"]["position"]).to_numpy(),
                    ak.flatten(tracks["steps"]["time"]).to_numpy()[:, None],
                ],
                axis=1,
            )

            energy = ak.flatten(tracks["steps"]["deposited_energy"]).to_numpy()
            vertex_positions, energy = voxelise_points(vertex_positions, self.edges, energy)
            energy_selector = energy > self.min_energy
            vertex_positions = vertex_positions[energy_selector]
            energy = energy[energy_selector]

            if vertex_positions.shape[0] == 0:
                warn(f"No vertices found in event {entry['mc_index'].item()} in file {file_path}")

            energy_sort_i = np.argsort(-energy)
            energy = energy[energy_sort_i]
            vertex_positions = vertex_positions[energy_sort_i]
            exists = np.ones(energy.shape[0], dtype=bool)

            if len(energy_sort_i) > self.max_n_vertices:
                energy = energy[: self.max_n_vertices]
                vertex_positions = vertex_positions[: self.max_n_vertices]
                exists = exists[: self.max_n_vertices]

            pad_kwargs = dict(pad_length=self.max_n_vertices, axis=0, generator=self.generator)
            exists = pad_array(exists, **pad_kwargs)
            vertices = pad_array(vertex_positions, **pad_kwargs)
            energy = pad_array(energy, **pad_kwargs)

            vertices = {
                "position": vertices[:, :3],
                "time": vertices[:, 3],
                "energy": energy,
                "exists": exists,
            }

            for key, val in vertices.items():
                if not np.isfinite(val).all():
                    raise ValueError(f"Non-finite vertex {key} values")

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


class MultiHitDatasetUnique(MultiHitDatasetBase):
    """
    PMT representation: unique PMT IDs + per-PMT hit counts.

    Outputs ``pmt_ids`` and ``pmt_id_counts`` each of length ``n_pmts``,
    and ``hit_times`` of length ``max_context_len``.  Compatible with
    :class:`~deepsno.models.multihit.MultiHitEncoder`, which uses
    ``torch.segment_reduce`` to aggregate hit-time embeddings per PMT.
    """

    def _make_pmt_inputs(self, pmt_ids: np.ndarray, hit_times: np.ndarray) -> dict[str, np.ndarray]:
        uniq_pmt_ids, pmt_id_counts = np.unique(pmt_ids, return_counts=True)
        pad_tuple = (0, self.n_pmts - len(uniq_pmt_ids))
        uniq_pmt_ids = np.pad(uniq_pmt_ids, pad_tuple)
        pmt_id_counts = np.pad(pmt_id_counts, pad_tuple)
        return {
            "pmt_ids": uniq_pmt_ids,
            "pmt_id_counts": pmt_id_counts,
            "hit_times": hit_times,
        }


class MultiHitDatasetExpanded(MultiHitDatasetBase):
    """
    PMT representation: one entry per hit (expanded / repeated form).

    Outputs ``pmt_ids`` and ``hit_times`` each of length ``max_context_len``,
    sorted by PMT ID.  Suitable for models that process individual hits rather
    than aggregated per-PMT features.
    """

    def _make_pmt_inputs(self, pmt_ids: np.ndarray, hit_times: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "pmt_ids": pmt_ids,
            "hit_times": hit_times,
        }


# Backward-compatible alias
MultiHitDataset = MultiHitDatasetUnique


class MultiHitVertexDataset(UprootMultiFileDataset):
    """
    Dataset for the flat hit_times/hit_ids schema with raw vertex truth data.

    Unlike MultiHitDatasetBase, this reads hits from flat per-hit arrays rather
    than nested per-PMT arrays, and uses the 'vertices' branch directly instead
    of voxelising tracks.

    Outputs:
        inputs: pmt_ids, hit_times — each (max_context_len,), sorted by pmt_id
        truth:  position (max_n_vertices, 3), time (max_n_vertices,),
                energy (max_n_vertices,), exists (max_n_vertices,),
                mc_index, npe, file_path
    """

    def __init__(
        self,
        max_context_len: int,
        max_n_vertices: int,
        min_hit_time: float = 0.0,
        max_hit_time: float = 300.0,
        min_energy: float = 0.0,
        time_jitter_min: float = 0.0,
        time_jitter_max: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_context_len = max_context_len
        self.max_n_vertices = max_n_vertices
        self.min_hit_time = min_hit_time
        self.max_hit_time = max_hit_time
        self.min_energy = min_energy
        self.time_jitter_min = time_jitter_min
        self.time_jitter_max = time_jitter_max
        # filter_name with a regex loads vertices sub-branches and reconstructs nesting
        self.filter_name.update(["hit_times", "hit_ids", "mc_index", "npe", "/vertices\\..*/"])

    def __iter__(self):
        for entry, file_path in super().__iter__():
            hit_times = entry["hit_times"].to_numpy()
            pmt_ids = entry["hit_ids"].to_numpy()

            selector = (hit_times > self.min_hit_time) & (hit_times < self.max_hit_time)
            hit_times = hit_times[selector]
            pmt_ids = pmt_ids[selector]

            if self.time_jitter_min != self.time_jitter_max:
                jitter = self.generator.uniform(self.time_jitter_min, self.time_jitter_max)
                hit_times = hit_times + jitter
            else:
                jitter = 0.0

            if len(pmt_ids) > self.max_context_len:
                shuffle_indices = self.generator.choice(len(pmt_ids), size=self.max_context_len, replace=False)
                pmt_ids = pmt_ids[shuffle_indices]
                hit_times = hit_times[shuffle_indices]

            sort_idx = np.argsort(pmt_ids)
            pmt_ids = pmt_ids[sort_idx]
            hit_times = hit_times[sort_idx]

            inputs = {"pmt_ids": pmt_ids, "hit_times": hit_times}

            # Vertex truth — already the actual interaction vertices
            verts = entry["vertices"]
            positions = ak.to_numpy(verts["position"])  # (n_verts, 3)
            times = ak.to_numpy(verts["time"])  # (n_verts,)
            energies = ak.to_numpy(verts["energy"])  # (n_verts,)

            times = times + jitter

            energy_mask = energies > self.min_energy
            positions = positions[energy_mask]
            times = times[energy_mask]
            energies = energies[energy_mask]

            energy_sort_i = np.argsort(-energies)
            positions = positions[energy_sort_i]
            times = times[energy_sort_i]
            energies = energies[energy_sort_i]
            exists = np.ones(len(energies), dtype=bool)

            if len(energies) > self.max_n_vertices:
                positions = positions[: self.max_n_vertices]
                times = times[: self.max_n_vertices]
                energies = energies[: self.max_n_vertices]
                exists = exists[: self.max_n_vertices]

            pad_kwargs = dict(pad_length=self.max_n_vertices, axis=0, generator=self.generator)
            vertex_shuffle_i = self.generator.permutation(self.max_n_vertices)

            vertices = {
                "position": pad_array(positions, **pad_kwargs)[vertex_shuffle_i],
                "time": pad_array(times, **pad_kwargs)[vertex_shuffle_i],
                "energy": pad_array(energies, **pad_kwargs)[vertex_shuffle_i],
                "exists": pad_array(exists, **pad_kwargs)[vertex_shuffle_i],
            }

            truth = {
                **pytree.tree_map(torch.from_numpy, vertices),
                "mc_index": entry["mc_index"].item(),
                "npe": entry["npe"].item(),
                "file_path": file_path,
            }

            yield pytree.tree_map(torch.from_numpy, inputs), truth


class MultiHitVarlenCollate:
    """Callable collate class that converts padded per-item hit sequences into the
    flat varlen format expected by flash attention / varlen attention kernels.

    Zero-padded hits (where ``pmt_ids == 0``) are stripped from each sequence.
    All tensors whose leading dimension matches the hit sequence length are
    concatenated into flat ``(total_hits,)`` tensors; other tensors (e.g.
    ``pmt_id_counts`` in :class:`MultiHitDatasetUnique`) are stacked normally.

    The returned ``inputs`` dict contains ``cu_seqlens`` (Int32, shape
    ``(B+1,)``) and ``max_seqlen`` (int) alongside the flat hit tensors.

    Usage in config::

        collate_fn:
            class_path: deepsno.data.multihit.MultiHitVarlenCollate
    """

    def __call__(self, batch: list) -> tuple[dict, dict]:
        inputs_list, truth_list = zip(*batch)

        seqlens: list[int] = []
        masked_keys: set[str] | None = None
        accum: dict[str, list[torch.Tensor]] = {}

        for inputs in inputs_list:
            mask = inputs["pmt_ids"] != 0
            n_valid = int(mask.sum())
            seqlens.append(n_valid)
            if masked_keys is None:
                masked_keys = {key for key, val in inputs.items() if val.shape == mask.shape}
            for key, val in inputs.items():
                accum.setdefault(key, []).append(val[mask] if key in masked_keys else val)

        cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)

        collated_inputs = {
            key: (torch.cat(vals) if key in masked_keys else torch.stack(vals)) for key, vals in accum.items()
        }
        collated_inputs["cu_seqlens"] = cu_seqlens
        collated_inputs["max_seqlen"] = max(seqlens)

        return collated_inputs, torch.utils.data.default_collate(list(truth_list))
