import numpy as np
import pytest

from deepsno.data.multihit import voxelise_points


@pytest.fixture
def uniform_edges():
    return [np.linspace(0, 10, 11), np.linspace(0, 10, 11)]  # 10 bins per axis, spacing=1


def test_basic_positions(uniform_edges):
    points = np.array([[0.5, 0.5], [1.5, 2.5]], dtype=np.float32)
    verts = voxelise_points(points, uniform_edges)
    expected = np.array([[0.5, 0.5], [1.5, 2.5]], dtype=np.float32)
    np.testing.assert_allclose(verts, expected)


def test_out_of_bounds_filtered(uniform_edges):
    points = np.array([[-1.0, 0.5], [0.5, 0.5], [100.0, 0.5]], dtype=np.float32)
    verts = voxelise_points(points, uniform_edges)
    assert verts.shape[0] == 1
    np.testing.assert_allclose(verts[0], [0.5, 0.5])


def test_duplicate_points_deduplicated(uniform_edges):
    # Three points in the same voxel → one vertex
    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.8, 0.9]], dtype=np.float32)
    verts = voxelise_points(points, uniform_edges)
    assert verts.shape[0] == 1
    np.testing.assert_allclose(verts[0], [0.5, 0.5])


def test_aux_values_summed(uniform_edges):
    # Two points in one voxel, one in another
    points = np.array([[0.1, 0.1], [0.9, 0.9], [5.5, 5.5]], dtype=np.float32)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    verts, reduced = voxelise_points(points, uniform_edges, weights)

    assert verts.shape[0] == 2
    # voxel [0,0]: sum of 1+2=3, voxel [5,5]: 3
    voxel_sums = dict(zip(map(tuple, verts.tolist()), reduced.tolist()))
    assert voxel_sums[(0.5, 0.5)] == pytest.approx(3.0)
    assert voxel_sums[(5.5, 5.5)] == pytest.approx(3.0)


def test_multiple_aux_values_independent(uniform_edges):
    # Regression test for the bug where the second aux value wasn't sorted correctly
    points = np.array([[5.5, 5.5], [0.1, 0.1], [0.9, 0.9]], dtype=np.float32)
    weights_a = np.array([10.0, 1.0, 2.0], dtype=np.float32)
    weights_b = np.array([100.0, 10.0, 20.0], dtype=np.float32)
    verts, reduced_a, reduced_b = voxelise_points(points, uniform_edges, weights_a, weights_b)

    assert verts.shape[0] == 2
    sums_a = dict(zip(map(tuple, verts.tolist()), reduced_a.tolist()))
    sums_b = dict(zip(map(tuple, verts.tolist()), reduced_b.tolist()))

    assert sums_a[(0.5, 0.5)] == pytest.approx(3.0)   # 1+2
    assert sums_a[(5.5, 5.5)] == pytest.approx(10.0)

    assert sums_b[(0.5, 0.5)] == pytest.approx(30.0)  # 10+20
    assert sums_b[(5.5, 5.5)] == pytest.approx(100.0)


def test_no_aux_returns_array(uniform_edges):
    points = np.array([[2.5, 3.5]], dtype=np.float32)
    result = voxelise_points(points, uniform_edges)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)


def test_all_out_of_bounds(uniform_edges):
    points = np.array([[-5.0, -5.0], [20.0, 20.0]], dtype=np.float32)
    verts = voxelise_points(points, uniform_edges)
    assert verts.shape[0] == 0
