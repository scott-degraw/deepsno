import glob
from abc import ABC, abstractmethod
from typing import Iterable

import awkward as ak
import fsspec
import numba as nb
import numpy as np
import torch
import torch.distributed as dist
import uproot
from torch.utils import _pytree as pytree
from torch.utils.data import IterableDataset

from deepsno.models.transformers import VarlenTensor


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

    pad_widths = [(0, 0)] * array.ndim
    pad_widths[axis] = (0, pad_width)
    return np.pad(array, pad_widths)


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
        active = voxelise_track(positions, edges, active=active)

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
        sort_i = np.argsort(uniq_2_non_uniq_indices)
        sorted_inverse = uniq_2_non_uniq_indices[sort_i]
        _, uniq_label_start_indices = np.unique(sorted_inverse, return_index=True)

        reduced_aux_values = [
            np.add.reduceat(value[good_indices][sort_i], uniq_label_start_indices) for value in aux_values
        ]

        return vertex_positions, *reduced_aux_values
    return vertex_positions


@nb.njit(cache=True)
def _rdp_compress_tracks_nb(
    points: np.ndarray,
    energies: np.ndarray,
    init_lo: np.ndarray,
    init_hi: np.ndarray,
    epsilon_xyz: float,
    epsilon_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch RDP over concatenated tracks.

    init_lo/init_hi are one entry per track; the algorithm never crosses those boundaries.
    Energy from dropped points is forward-aggregated within each track only.
    """
    n = len(points)
    n_tracks = len(init_lo)
    inv_xyz = 1.0 / epsilon_xyz
    inv_t = 1.0 / epsilon_t

    # Scale into normalised space so the distance threshold is uniformly 1.0
    scaled = np.empty((n, 4), dtype=np.float32)
    for i in range(n):
        scaled[i, 0] = points[i, 0] * inv_xyz
        scaled[i, 1] = points[i, 1] * inv_xyz
        scaled[i, 2] = points[i, 2] * inv_xyz
        scaled[i, 3] = points[i, 3] * inv_t

    # Mark all track endpoints as kept
    mask = np.zeros(n, dtype=np.bool_)
    for i in range(n_tracks):
        mask[init_lo[i]] = True
        mask[init_hi[i]] = True

    # Seed the stack with one (lo, hi) interval per track
    stack_cap = n + n_tracks
    stack_lo = np.empty(stack_cap, dtype=np.int64)
    stack_hi = np.empty(stack_cap, dtype=np.int64)
    for i in range(n_tracks):
        stack_lo[i] = init_lo[i]
        stack_hi[i] = init_hi[i]
    top = n_tracks

    while top > 0:
        top -= 1
        lo = stack_lo[top]
        hi = stack_hi[top]
        if hi - lo <= 1:
            continue

        s0 = scaled[hi, 0] - scaled[lo, 0]
        s1 = scaled[hi, 1] - scaled[lo, 1]
        s2 = scaled[hi, 2] - scaled[lo, 2]
        s3 = scaled[hi, 3] - scaled[lo, 3]
        seg_len_sq = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3
        max_dist = -1.0
        best = lo + 1
        for i in range(lo + 1, hi):
            d0 = scaled[i, 0] - scaled[lo, 0]
            d1 = scaled[i, 1] - scaled[lo, 1]
            d2 = scaled[i, 2] - scaled[lo, 2]
            d3 = scaled[i, 3] - scaled[lo, 3]
            if seg_len_sq < 1e-20:
                dist = (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3) ** 0.5
            else:
                tp = (d0 * s0 + d1 * s1 + d2 * s2 + d3 * s3) / seg_len_sq
                if tp < 0.0:
                    tp = 0.0
                elif tp > 1.0:
                    tp = 1.0
                r0 = d0 - tp * s0
                r1 = d1 - tp * s1
                r2 = d2 - tp * s2
                r3 = d3 - tp * s3
                dist = (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3) ** 0.5
            if dist > max_dist:
                max_dist = dist
                best = i

        if max_dist >= 1.0:
            mask[best] = True
            stack_lo[top] = lo
            stack_hi[top] = best
            top += 1
            stack_lo[top] = best
            stack_hi[top] = hi
            top += 1

    # Forward-aggregate dropped energies onto the next kept point within each track.
    # Track boundaries are always masked True, so aggregation never crosses tracks.
    agg_e = energies.copy()
    for k in range(n - 1):
        if not mask[k]:
            nxt = k + 1
            while nxt < n - 1 and not mask[nxt]:
                nxt += 1
            agg_e[nxt] += energies[k]

    # Gather kept points (boolean fancy-indexing not supported in njit)
    m = 0
    for i in range(n):
        if mask[i]:
            m += 1
    kept_points = np.empty((m, 4), dtype=points.dtype)
    kept_energies = np.empty(m, dtype=energies.dtype)
    j = 0
    for i in range(n):
        if mask[i]:
            kept_points[j, 0] = points[i, 0]
            kept_points[j, 1] = points[i, 1]
            kept_points[j, 2] = points[i, 2]
            kept_points[j, 3] = points[i, 3]
            kept_energies[j] = agg_e[i]
            j += 1

    return kept_points, kept_energies


def rdp_compress_track(
    points: np.ndarray,
    energies: np.ndarray,
    epsilon_xyz: float,
    epsilon_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """RDP simplification for a single (N, 4) [x,y,z,t] track with energy aggregation.

    Normalises coordinates by (epsilon_xyz, epsilon_t) so the perpendicular-distance
    threshold of 1 applies uniformly to both spaces.  Energies of removed points are
    forward-accumulated onto the next kept point in the sequence.

    Args:
        points: (N, 4) array of [x, y, z, t] midpoints.
        energies: (N,) deposited energies corresponding to each midpoint.
        epsilon_xyz: Spatial simplification tolerance (same units as points[:, :3]).
        epsilon_t: Temporal simplification tolerance (same units as points[:, 3]).

    Returns:
        kept_points: (M, 4) simplified polyline, M <= N.
        kept_energies: (M,) aggregated energies for the kept points.
    """
    n = len(points)
    if n < 3:
        return points.copy(), energies.copy()
    pts = np.asarray(points, dtype=np.float32)
    e = np.asarray(energies, dtype=np.float32)
    init_lo = np.array([0], dtype=np.int64)
    init_hi = np.array([n - 1], dtype=np.int64)
    return _rdp_compress_tracks_nb(
        points=pts,
        energies=e,
        init_lo=init_lo,
        init_hi=init_hi,
        epsilon_xyz=float(epsilon_xyz),
        epsilon_t=float(epsilon_t),
    )


# ---------------------------------------------------------------------------
# Hit input makers
# ---------------------------------------------------------------------------


class HitInputMaker(ABC):
    """Produces the model input dict from a single ROOT entry."""

    @property
    @abstractmethod
    def filter_names(self) -> set[str]:
        """Uproot branch names / patterns this maker needs."""
        ...

    @abstractmethod
    def make_inputs(
        self,
        entry,
        generator: np.random.Generator,
        max_context_len: int,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Return ``(inputs, context)``.

        ``inputs`` maps string keys to numpy arrays that will be converted to
        tensors and yielded by the dataset.  ``context`` is an optional dict of
        shared values (e.g. time jitter) forwarded to the vertex maker.
        """
        ...


class NestedHitInputMaker(HitInputMaker):
    """Reads nested per-PMT ``hit_times`` arrays (the standard multihit schema).

    Subclasses can override ``_make_pmt_inputs`` to choose how the filtered,
    sorted arrays are represented (expanded vs. unique-PMT).
    """

    def __init__(
        self,
        min_hit_time: float = 0.0,
        max_hit_time: float = 300.0,
        pmt_info_path: str | None = None,
        active_pmt_types: tuple[str, ...] = ("NORMAL", "HQE"),
    ):
        self.min_hit_time = min_hit_time
        self.max_hit_time = max_hit_time
        if pmt_info_path is not None:
            from deepsno.data.pmt_info import active_pmt_remap
            active_ids, remap = active_pmt_remap(pmt_info_path, active_pmt_types)
            self._remap: np.ndarray | None = remap
            self._n_active: int = len(active_ids)
        else:
            self._remap = None
            self._n_active = None

    @property
    def filter_names(self) -> set[str]:
        return {"hit_times"}

    def _make_pmt_inputs(self, pmt_ids: np.ndarray, hit_times: np.ndarray) -> dict[str, np.ndarray]:
        return {"pmt_ids": pmt_ids, "hit_times": hit_times}

    def make_inputs(
        self,
        entry,
        generator: np.random.Generator,
        max_context_len: int,
    ) -> tuple[dict[str, np.ndarray], dict]:
        hits_per_pmt = ak.num(entry["hit_times"]).to_numpy()
        pmt_ids = np.arange(len(entry["hit_times"]))
        pmt_ids = np.repeat(pmt_ids, hits_per_pmt)
        hit_times = ak.flatten(entry["hit_times"]).to_numpy().astype(np.float32)

        selector = (hit_times > self.min_hit_time) & (hit_times < self.max_hit_time)
        pmt_ids = pmt_ids[selector]
        hit_times = hit_times[selector]

        if self._remap is not None:
            compact = self._remap[pmt_ids]
            keep = compact >= 0
            pmt_ids = compact[keep]
            hit_times = hit_times[keep]

        if np.any(~np.isfinite(np.log(hit_times))):
            raise ValueError("Bad log hit times")

        if len(pmt_ids) > max_context_len:
            idx = generator.choice(len(pmt_ids), size=max_context_len, replace=False)
            pmt_ids = pmt_ids[idx]
            hit_times = hit_times[idx]

        sort_idx = np.argsort(pmt_ids)
        pmt_ids = pmt_ids[sort_idx]
        hit_times = hit_times[sort_idx]

        return self._make_pmt_inputs(pmt_ids, hit_times), {}


class ExpandedHitInputMaker(NestedHitInputMaker):
    """One entry per hit: outputs ``pmt_ids`` and ``hit_times`` of length ``max_context_len``."""

    pass


class UniqueHitInputMaker(NestedHitInputMaker):
    """Per-PMT aggregation: flat hit times sorted by PMT id + a fixed-size count array.

    Outputs:
        hit_times:   ``(n_hits,)`` — unpadded, sorted by pmt_id.
        pmt_lengths: ``(n_pmts,)`` — hit count per PMT slot (0 for inactive PMTs).

    Designed for use with :class:`MultiHitUniqueCollate` which concatenates
    ``hit_times`` across the batch and stacks ``pmt_lengths`` to ``(B, n_pmts)``.
    The encoder uses ``torch.segment_reduce`` to aggregate hits per PMT, producing
    a fixed-size ``(B, n_pmts, dim)`` representation.
    """

    def __init__(self, n_pmts: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if n_pmts is not None:
            self.n_pmts = n_pmts
        elif self._n_active is not None:
            self.n_pmts = self._n_active
        else:
            raise ValueError("UniqueHitInputMaker requires either n_pmts or pmt_info_path")

    def _make_pmt_inputs(self, pmt_ids: np.ndarray, hit_times: np.ndarray) -> dict[str, np.ndarray]:
        pmt_lengths = np.bincount(pmt_ids, minlength=self.n_pmts).astype(np.int64)
        return {"hit_times": hit_times, "pmt_lengths": pmt_lengths}


class FlatHitInputMaker(HitInputMaker):
    """Reads flat per-hit ``hit_times`` / ``hit_ids`` arrays (the vertex dataset schema).

    Puts the sampled ``jitter`` value into the context dict so that
    :class:`RawVertexMaker` can apply the same shift to vertex times.
    """

    def __init__(
        self,
        min_hit_time: float = 0.0,
        max_hit_time: float = 300.0,
        time_jitter_min: float = 0.0,
        time_jitter_max: float = 0.0,
    ):
        self.min_hit_time = min_hit_time
        self.max_hit_time = max_hit_time
        self.time_jitter_min = time_jitter_min
        self.time_jitter_max = time_jitter_max

    @property
    def filter_names(self) -> set[str]:
        return {"hit_times", "hit_ids"}

    def make_inputs(
        self,
        entry,
        generator: np.random.Generator,
        max_context_len: int,
    ) -> tuple[dict[str, np.ndarray], dict]:
        hit_times = entry["hit_times"].to_numpy()
        pmt_ids = entry["hit_ids"].to_numpy()

        selector = (hit_times > self.min_hit_time) & (hit_times < self.max_hit_time)
        hit_times = hit_times[selector]
        pmt_ids = pmt_ids[selector]

        if self.time_jitter_min != self.time_jitter_max:
            jitter = float(generator.uniform(self.time_jitter_min, self.time_jitter_max))
        else:
            jitter = 0.0
        hit_times = hit_times + jitter

        if len(pmt_ids) > max_context_len:
            idx = generator.choice(len(pmt_ids), size=max_context_len, replace=False)
            pmt_ids = pmt_ids[idx]
            hit_times = hit_times[idx]

        sort_idx = np.argsort(pmt_ids)
        pmt_ids = pmt_ids[sort_idx]
        hit_times = hit_times[sort_idx]

        return {"pmt_ids": pmt_ids, "hit_times": hit_times}, {"jitter": jitter}


# ---------------------------------------------------------------------------
# Vertex data makers
# ---------------------------------------------------------------------------


class VertexDataMaker(ABC):
    """Produces the vertex truth dict from a single ROOT entry."""

    @property
    @abstractmethod
    def filter_names(self) -> set[str]:
        """Uproot branch names / patterns this maker needs."""
        ...

    @abstractmethod
    def make_truth(
        self,
        entry,
        generator: np.random.Generator,
        max_n_vertices: int,
        context: dict,
    ) -> dict[str, np.ndarray] | None:
        """Return a dict of numpy arrays keyed by vertex field name, or ``None`` to skip the event.

        The returned dict should contain ``position``, ``time``, ``energy``,
        and ``exists`` as numpy arrays of length ``max_n_vertices``.  The
        dataset adds ``mc_index``, ``npe``, and ``file_path`` separately.
        """
        ...


def _pad_and_shuffle_vertices(
    vertex_positions: np.ndarray,
    energy: np.ndarray,
    exists: np.ndarray,
    max_n_vertices: int,
    generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Pad / truncate vertex arrays to ``max_n_vertices`` and randomly shuffle."""
    pad_kwargs = dict(pad_length=max_n_vertices, axis=0, generator=generator)
    vertices = pad_array(vertex_positions, **pad_kwargs)
    energy = pad_array(energy, **pad_kwargs)
    exists = pad_array(exists, **pad_kwargs)

    shuffle_i = generator.permutation(max_n_vertices)
    return {
        "position": vertices[shuffle_i, :3],
        "time": vertices[shuffle_i, 3],
        "energy": energy[shuffle_i],
        "exists": exists[shuffle_i],
    }


class VoxelizedTrackVertexMaker(VertexDataMaker):
    """Builds vertex truth by voxelising Geant4 track step positions.

    Track step positions are discretised onto a 4-D (x, y, z, t) grid and
    energies accumulated per voxel.  The top ``max_n_vertices`` voxels by
    energy are returned as the vertex set.
    """

    def __init__(
        self,
        radius: float,
        pos_spacing: float,
        time_spacing: float,
        time_low: float = 0.0,
        time_high: float = 40.0,
        min_energy: float = 0.0,
        step_energy_key: str = "n_photons",
    ):
        self.min_energy = min_energy
        self.step_energy_key = step_energy_key

        self.edges = [np.arange(-radius, radius + pos_spacing, pos_spacing, dtype=np.float32) for _ in range(3)]
        self.edges += [np.arange(time_low, time_high + time_spacing, time_spacing, dtype=np.float32)]

    @property
    def filter_names(self) -> set[str]:
        return {"tracks*"}

    def make_truth(
        self,
        entry,
        generator: np.random.Generator,
        max_n_vertices: int,
        context: dict,
    ) -> dict[str, np.ndarray] | None:
        tracks = entry["tracks"]

        vertex_positions = np.concatenate(
            [
                ak.flatten(tracks["steps"]["position"]).to_numpy(),
                ak.flatten(tracks["steps"]["time"]).to_numpy()[:, None],
            ],
            axis=1,
        )

        _energy_key = self.step_energy_key if self.step_energy_key in tracks["steps"].fields else "deposited_energy"
        energy = ak.flatten(tracks["steps"][_energy_key]).to_numpy().astype(np.float32)
        vertex_positions, energy = voxelise_points(vertex_positions, self.edges, energy)

        energy_selector = energy > self.min_energy
        vertex_positions = vertex_positions[energy_selector]
        energy = energy[energy_selector]

        if vertex_positions.shape[0] == 0:
            return None

        energy_sort_i = np.argsort(-energy)
        energy = energy[energy_sort_i]
        vertex_positions = vertex_positions[energy_sort_i]
        exists = np.ones(energy.shape[0], dtype=bool)

        if len(energy) > max_n_vertices:
            energy = energy[:max_n_vertices]
            vertex_positions = vertex_positions[:max_n_vertices]
            exists = exists[:max_n_vertices]

        result = _pad_and_shuffle_vertices(
            vertex_positions=vertex_positions,
            energy=energy,
            exists=exists,
            max_n_vertices=max_n_vertices,
            generator=generator,
        )

        for key, val in result.items():
            if not np.isfinite(val).all():
                raise ValueError(f"Non-finite vertex {key} values")

        return result


class RDPTrackVertexMaker(VertexDataMaker):
    """Builds vertex truth from RDP-compressed track step midpoints.

    Midpoints of consecutive Geant4 steps are computed per track, optionally
    filtered by time / radius / energy, then compressed with the
    Ramer-Douglas-Peucker algorithm.  Energies of removed points are
    forward-accumulated onto the next kept point within each track.
    """

    def __init__(
        self,
        epsilon_xyz: float,
        epsilon_t: float,
        min_energy: float = 0.0,
        max_energy: float | None = None,
        max_step_time: float | None = None,
        max_step_radius: float | None = None,
        step_energy_key: str = "n_photons",
    ):
        self.epsilon_xyz = epsilon_xyz
        self.epsilon_t = epsilon_t
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.max_step_time = max_step_time
        self.max_step_radius = max_step_radius
        self.step_energy_key = step_energy_key

    @property
    def filter_names(self) -> set[str]:
        return {"tracks*"}

    def make_truth(
        self,
        entry,
        generator: np.random.Generator,
        max_n_vertices: int,
        context: dict,
    ) -> dict[str, np.ndarray] | None:
        all_tracks = entry["tracks"]
        _energy_key = self.step_energy_key if self.step_energy_key in all_tracks["steps"].fields else "deposited_energy"
        track_energy = ak.sum(all_tracks["steps"][_energy_key], axis=1)
        tracks = all_tracks[track_energy > self.min_energy]

        pos_flat = ak.flatten(tracks["steps"]["position"]).to_numpy()  # (N, 3)
        t_flat = ak.flatten(tracks["steps"]["time"]).to_numpy()  # (N,)
        e_flat = ak.flatten(tracks["steps"][_energy_key]).to_numpy().astype(np.float32)  # (N,)
        steps_per_track = ak.num(tracks["steps"]["time"]).to_numpy()  # (T,)

        if len(pos_flat) >= 2:
            track_id_per_step = np.repeat(np.arange(len(steps_per_track)), steps_per_track)
            valid_pair = track_id_per_step[:-1] == track_id_per_step[1:]

            mid_pos = 0.5 * (pos_flat[:-1][valid_pair] + pos_flat[1:][valid_pair])  # (M, 3)
            mid_t = 0.5 * (t_flat[:-1][valid_pair] + t_flat[1:][valid_pair])  # (M,)
            mid_e = e_flat[1:][valid_pair]  # (M,)

            midpoints_per_track = np.maximum(steps_per_track - 1, 0)

            filt = np.ones(len(mid_e), dtype=bool)
            if self.max_step_time is not None:
                filt &= mid_t <= self.max_step_time
            if self.max_step_radius is not None:
                filt &= np.linalg.norm(mid_pos, axis=1) <= self.max_step_radius
            if self.max_energy is not None:
                filt &= mid_e <= self.max_energy

            if not np.all(filt):
                mid_track_id = np.repeat(np.arange(len(steps_per_track)), midpoints_per_track)
                midpoints_per_track = np.bincount(mid_track_id[filt], minlength=len(steps_per_track))
                mid_pos = mid_pos[filt]
                mid_t = mid_t[filt]
                mid_e = mid_e[filt]
        else:
            mid_e = np.zeros(0, dtype=np.float32)
            midpoints_per_track = np.zeros(0, dtype=np.int64)

        if len(mid_e) == 0:
            step_pts = np.zeros((0, 4), dtype=np.float32)
            step_e = np.zeros(0, dtype=np.float32)
        else:
            pts_4d = np.concatenate([mid_pos, mid_t[:, None]], axis=1).astype(np.float32)

            mid_cumsum = np.concatenate([[0], np.cumsum(midpoints_per_track)])
            has_mids = midpoints_per_track >= 1
            init_lo = mid_cumsum[:-1][has_mids].astype(np.int64)
            init_hi = (mid_cumsum[1:][has_mids] - 1).astype(np.int64)

            step_pts, step_e = _rdp_compress_tracks_nb(
                points=pts_4d,
                energies=mid_e.astype(np.float32),
                init_lo=init_lo,
                init_hi=init_hi,
                epsilon_xyz=float(self.epsilon_xyz),
                epsilon_t=float(self.epsilon_t),
            )
            step_pts = step_pts.astype(np.float32)
            step_e = step_e.astype(np.float32)

        energy_sel = step_e > self.min_energy
        step_pts = step_pts[energy_sel]
        step_e = step_e[energy_sel]

        if len(step_e) == 0:
            return None

        sort_i = np.argsort(-step_e)
        step_pts = step_pts[sort_i]
        step_e = step_e[sort_i]
        exists = np.ones(len(step_e), dtype=bool)

        if len(step_e) > max_n_vertices:
            step_pts = step_pts[:max_n_vertices]
            step_e = step_e[:max_n_vertices]
            exists = exists[:max_n_vertices]

        result = _pad_and_shuffle_vertices(
            vertex_positions=step_pts,
            energy=step_e,
            exists=exists,
            max_n_vertices=max_n_vertices,
            generator=generator,
        )

        for key, val in result.items():
            if not np.isfinite(val).all():
                raise ValueError(f"Non-finite vertex {key} values")

        return result


class RawVertexMaker(VertexDataMaker):
    """Reads vertex truth directly from a ``vertices`` branch (no track voxelisation).

    Applies the same time jitter that :class:`FlatHitInputMaker` put into
    ``context["jitter"]``, so hits and vertices remain consistent.
    """

    def __init__(self, min_energy: float = 0.0):
        self.min_energy = min_energy

    @property
    def filter_names(self) -> set[str]:
        return {"/vertices\\..*/"}

    def make_truth(
        self,
        entry,
        generator: np.random.Generator,
        max_n_vertices: int,
        context: dict,
    ) -> dict[str, np.ndarray] | None:
        jitter = context.get("jitter", 0.0)

        verts = entry["vertices"]
        positions = ak.to_numpy(verts["position"])  # (n_verts, 3)
        times = ak.to_numpy(verts["time"]) + jitter  # (n_verts,)
        energies = ak.to_numpy(verts["energy"])  # (n_verts,)

        energy_mask = energies > self.min_energy
        positions = positions[energy_mask]
        times = times[energy_mask]
        energies = energies[energy_mask]

        if len(energies) == 0:
            return None

        energy_sort_i = np.argsort(-energies)
        positions = positions[energy_sort_i]
        times = times[energy_sort_i]
        energies = energies[energy_sort_i]
        exists = np.ones(len(energies), dtype=bool)

        if len(energies) > max_n_vertices:
            positions = positions[:max_n_vertices]
            times = times[:max_n_vertices]
            energies = energies[:max_n_vertices]
            exists = exists[:max_n_vertices]

        vertex_positions = np.concatenate([positions, times[:, None]], axis=1)
        return _pad_and_shuffle_vertices(
            vertex_positions=vertex_positions,
            energy=energies,
            exists=exists,
            max_n_vertices=max_n_vertices,
            generator=generator,
        )


# ---------------------------------------------------------------------------
# Unified composable dataset
# ---------------------------------------------------------------------------


class MultiHitDataset(UprootMultiFileDataset):
    """Composable dataset that delegates hit-input and vertex-truth building to separate classes.

    Args:
        hit_input_maker: Produces the model input dict for each event.
        vertex_data_maker: Produces the vertex truth dict for each event.
        max_context_len: Maximum number of hits per event (truncated / sub-sampled if exceeded).
        max_n_vertices: Maximum number of vertices per event (truncated / padded).
        All remaining kwargs are forwarded to :class:`UprootMultiFileDataset`.
    """

    def __init__(
        self,
        hit_input_maker: HitInputMaker,
        vertex_data_maker: VertexDataMaker,
        max_context_len: int,
        max_n_vertices: int,
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
    ):
        combined_filter = set(filter_name) if filter_name else set()
        combined_filter |= {"mc_index", "npe"}
        combined_filter |= hit_input_maker.filter_names
        combined_filter |= vertex_data_maker.filter_names

        super().__init__(
            file_paths=file_paths,
            tree_name=tree_name,
            expressions=expressions,
            filter_name=combined_filter,
            cut=cut,
            seed=seed,
            buffer_size=buffer_size,
            shuffle=shuffle,
            cache=cache,
            debug=debug,
        )
        self.hit_input_maker = hit_input_maker
        self.vertex_data_maker = vertex_data_maker
        self.max_context_len = max_context_len
        self.max_n_vertices = max_n_vertices

    def __iter__(self):
        for entry, file_path in super().__iter__():
            inputs, context = self.hit_input_maker.make_inputs(
                entry, generator=self.generator, max_context_len=self.max_context_len
            )
            if inputs is None:
                continue

            vertex_data = self.vertex_data_maker.make_truth(
                entry, generator=self.generator, max_n_vertices=self.max_n_vertices, context=context
            )
            if vertex_data is None:
                continue

            truth = {
                **pytree.tree_map(torch.from_numpy, vertex_data),
                "mc_index": entry["mc_index"].item(),
                "npe": entry["npe"].item(),
                "file_path": file_path,
            }

            yield pytree.tree_map(torch.from_numpy, inputs), truth


MultiHitDatasetExpanded = MultiHitDataset


class MultiHitUniqueCollate:
    """Collate for :class:`UniqueHitInputMaker` output.

    Concatenates the unpadded ``hit_times`` arrays across the batch and stacks
    ``pmt_lengths`` to ``(B, n_pmts)``.  No varlen bookkeeping is needed here —
    the encoder performs ``segment_reduce`` directly on the flat hit sequence.

    Usage in config::

        collate_fn:
            class_path: deepsno.data.multihit.MultiHitUniqueCollate
    """

    def __call__(self, batch: list) -> tuple[dict, dict]:
        inputs_list, truth_list = zip(*batch)
        collated_inputs = {
            "hit_times": torch.cat([x["hit_times"] for x in inputs_list]),
            "pmt_lengths": torch.stack([x["pmt_lengths"] for x in inputs_list]),
        }
        return collated_inputs, torch.utils.data.default_collate(list(truth_list))


class MultiHitVarlenCollate:
    """Callable collate class that converts padded per-item hit sequences into the
    flat varlen format expected by flash attention / varlen attention kernels.

    Zero-padded hits (where ``pmt_ids == 0``) are stripped from each sequence.
    All tensors whose leading dimension matches the hit sequence length are
    concatenated into flat ``(total_hits,)`` tensors; other tensors are stacked normally.

    The returned ``inputs`` dict contains a ``hits`` key holding a
    :class:`~deepsno.models.transformers.VarlenTensor` that bundles the flat
    ``pmt_ids``, ``cu_seqlens`` (Int32, ``(B+1,)``), and ``max_seqlen`` (int).
    All remaining flat tensors (e.g. ``hit_times``) are included as top-level keys.

    Usage in config::

        collate_fn:
            class_path: deepsno.data.multihit.MultiHitVarlenCollate
    """

    def __call__(self, batch: list) -> tuple[dict, dict]:
        inputs_list, truth_list = zip(*batch)

        seqlens: list[int] = []
        # The masked keys are those that have the same shape as the hit sequence and need to be masked and concatenated
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

        # concatenate the hit level inputs but stack and retain the rectangular structure otherwise
        flat = {key: (torch.cat(vals) if key in masked_keys else torch.stack(vals)) for key, vals in accum.items()}
        pmt_ids = flat.pop("pmt_ids")
        collated_inputs = {"hits": VarlenTensor(pmt_ids, cu_seqlens, max(seqlens)), **flat}

        return collated_inputs, torch.utils.data.default_collate(list(truth_list))
