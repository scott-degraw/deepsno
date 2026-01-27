import numpy as np
import plotly.graph_objects as go


def get_cube_mesh(
    x_center: np.ndarray,
    y_center: np.ndarray,
    z_center: np.ndarray,
    x_size: float = 1,
    y_size: float = 1,
    z_size: float = 1,
    **mesh3d_kwargs,
):
    # Calculate half-size for offsets
    sx = x_size / 2
    sy = y_size / 2
    sz = z_size / 2

    if not (x_center.shape == y_center.shape == z_center.shape):
        raise ValueError("x_center, y_center, and z_center must have the same shape")

    x_center = x_center.ravel()
    y_center = y_center.ravel()
    z_center = z_center.ravel()

    xr = x_center + sx
    xl = x_center - sx
    yr = y_center + sy
    yl = y_center - sy
    zr = z_center + sz
    zl = z_center - sz

    x_vertices = np.empty_like(x_center)
    x_vertices = np.repeat(x_vertices[:, None], repeats=8, axis=1)
    y_vertices = np.empty_like(y_center)
    y_vertices = np.repeat(y_vertices[:, None], repeats=8, axis=1)
    z_vertices = np.empty_like(z_center)
    z_vertices = np.repeat(z_vertices[:, None], repeats=8, axis=1)

    x_vertices[:, np.array([0, 1, 4, 5])] = xl[:, None]
    x_vertices[:, [2, 3, 6, 7]] = xr[:, None]

    y_vertices[:, [0, 3, 4, 7]] = yl[:, None]
    y_vertices[:, [1, 2, 5, 6]] = yr[:, None]

    z_vertices[:, [0, 1, 2, 3]] = zl[:, None]
    z_vertices[:, [4, 5, 6, 7]] = zr[:, None]

    # The 12 triangles that make up the 6 faces of a cube
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    meshes = []
    for x, y, z in zip(x_vertices, y_vertices, z_vertices):
        meshes.append(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                alphahull=-1,
                flatshading=True,
                **mesh3d_kwargs,
            )
        )

    return meshes