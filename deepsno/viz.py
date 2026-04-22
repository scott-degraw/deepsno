import awkward as ak
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

try:
    from particle import Particle

    def particle_name(pdg: int) -> str:
        try:
            return Particle.from_pdgid(pdg).name
        except Exception:
            return f"PDG {pdg}"

except ImportError:
    def particle_name(pdg: int) -> str:
        return f"PDG {pdg}"


COLORSCALE = "Plasma"
VOXEL_COLORSCALE = "Viridis"

_CUBE_VERT_OFFSETS = np.array([
    [-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
    [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1],
], dtype=float)

_CUBE_FACES = np.array([
    [0,1,2],[0,2,3],
    [4,5,6],[4,6,7],
    [0,1,5],[0,5,4],
    [2,3,7],[2,7,6],
    [0,3,7],[0,7,4],
    [1,2,6],[1,6,5],
])


def plot_tracks(tracks, t_min: float, t_max: float) -> list:
    traces = []
    t_range = t_max - t_min or 1.0
    for track in tracks:
        steps = track["steps"]
        pos = ak.to_numpy(steps["position"])
        t = ak.to_numpy(steps["time"])
        dep_step = ak.to_numpy(steps["deposited_energy"])
        dep_total = float(ak.sum(steps["deposited_energy"]))
        name = particle_name(int(track["pdg"]))
        colors = pc.sample_colorscale(COLORSCALE, (t - t_min) / t_range)
        traces.append(
            go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="lines",
                name=f"{name} ({dep_total:.2f} MeV dep.)",
                line=dict(width=6, color=colors),
                text=[f"t={tt:.1f} ns<br>dep={d:.3f} MeV" for tt, d in zip(t, dep_step)],
                hovertemplate="%{text}<extra>" + name + "</extra>",
            )
        )
    return traces


def plot_voxels(tracks, voxel_size: float = 100.0) -> list:
    voxel_energy: dict[tuple, float] = {}
    for track in tracks:
        steps = track["steps"]
        pos = ak.to_numpy(steps["position"])
        dep = ak.to_numpy(steps["deposited_energy"])
        for p, d in zip(pos, dep):
            if d <= 0:
                continue
            key = tuple((p // voxel_size).astype(int))
            voxel_energy[key] = voxel_energy.get(key, 0.0) + d

    if not voxel_energy:
        return []

    keys = np.array(list(voxel_energy.keys()))
    centers = (keys + 0.5) * voxel_size
    energies = np.array(list(voxel_energy.values()))
    n = len(centers)

    verts = (centers[:, np.newaxis, :] + _CUBE_VERT_OFFSETS[np.newaxis, :, :] * (voxel_size / 2)).reshape(-1, 3)
    face_offsets = (np.arange(n) * 8)[:, np.newaxis, np.newaxis]
    faces = (_CUBE_FACES[np.newaxis, :, :] + face_offsets).reshape(-1, 3)
    intensity = np.repeat(energies, 8)
    hover_text = np.repeat([f"dep={e:.4f} MeV" for e in energies], 8)

    return [
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=intensity,
            colorscale=VOXEL_COLORSCALE,
            opacity=0.35,
            showscale=True,
            colorbar=dict(title="Voxel dep. energy (MeV)", x=1.12),
            name=f"Voxels ({voxel_size:.0f} mm)",
            text=hover_text,
            hovertemplate="%{text}<extra>Voxel</extra>",
        )
    ]


def plot_truth_vertices(vertices, t_min: float, t_max: float) -> list:
    pos = ak.to_numpy(vertices["position"])
    t = ak.to_numpy(vertices["time"])
    energy = ak.to_numpy(vertices["energy"])
    pdgs = ak.to_numpy(vertices["pdg"])

    by_species: dict[str, list[int]] = {}
    for i, pdg in enumerate(pdgs):
        name = particle_name(int(pdg))
        by_species.setdefault(name, []).append(i)

    traces = []
    for species, indices in by_species.items():
        idx = np.array(indices)
        show_scale = species == next(iter(by_species))
        traces.append(
            go.Scatter3d(
                x=pos[idx, 0], y=pos[idx, 1], z=pos[idx, 2],
                mode="markers",
                name=f"Vertex: {species}",
                marker=dict(
                    size=10,
                    color=t[idx],
                    symbol="diamond",
                    line=dict(width=1, color="black"),
                    colorscale=COLORSCALE,
                    cmin=t_min,
                    cmax=t_max,
                    showscale=show_scale,
                    colorbar=dict(title="Time (ns)", x=1.0),
                ),
                text=[f"{species}<br>E={energy[i]:.2f} MeV<br>t={t[i]:.1f} ns" for i in idx],
                hovertemplate="%{text}<extra>Vertex</extra>",
            )
        )
    return traces
