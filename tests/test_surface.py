import numpy as np

from preprocess.SurfaceBuilder import SurfaceBuilder


def test_sphere_normals_curvature_connectivity_and_weights() -> None:
    radius = 5.0
    builder = SurfaceBuilder(0.7)
    points = builder.fibonacci_sphere(700) * radius
    normals = builder.estimate_normals(points, outward_reference=np.zeros(3))
    alignment = np.sum(normals * points / radius, axis=1)
    assert np.median(alignment) > 0.98
    curvature = builder.estimate_curvatures(points, normals)
    assert np.median(curvature[:, 1, 0]) > 0.12
    assert np.median(curvature[:, 1, 1]) > 0.01
    graph, warnings = builder.build_graph(points, normals)
    assert len(np.unique(graph["surface_component_ids"])) == 1
    assert not warnings
    weights = builder.area_weights(points)
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), 1.0)


def test_plane_normals_and_near_zero_curvature() -> None:
    axis = np.linspace(-4.0, 4.0, 17)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
    builder = SurfaceBuilder(0.5)
    normals = builder.estimate_normals(points, outward_reference=reference)
    assert np.min(normals[:, 2]) > 0.99
    curvature = builder.estimate_curvatures(points, normals)
    assert np.quantile(np.abs(curvature[:, :, 0]), 0.9) < 1.0e-5


def test_curvature_scale_count_and_order_are_configurable() -> None:
    axis = np.linspace(-2.0, 2.0, 9)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))

    curvature = SurfaceBuilder(
        resolution=0.5,
        curvature_scales=(1.5, 3.0, 6.0),
    ).estimate_curvatures(points, normals)

    assert curvature.shape == (len(points), 3, 3)
    assert np.isfinite(curvature).all()


def test_cylinder_normals_and_gaussian_curvature() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    heights = np.linspace(-4.0, 4.0, 17)
    theta, z = np.meshgrid(angles, heights)
    radius = 3.0
    points = np.column_stack(
        (radius * np.cos(theta.ravel()), radius * np.sin(theta.ravel()), z.ravel())
    )
    builder = SurfaceBuilder(0.5)
    normals = builder.estimate_normals(points, outward_reference=np.zeros(3))
    radial = points.copy()
    radial[:, 2] = 0.0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    interior = np.abs(points[:, 2]) < 3.5
    assert np.median(np.sum(normals[interior] * radial[interior], axis=1)) > 0.97
    curvature = builder.estimate_curvatures(points, normals)
    assert np.median(np.abs(curvature[interior, 0, 1])) < 0.03
    assert np.median(curvature[interior, 0, 2]) > 0.1


def test_concave_bowl_has_negative_mean_curvature() -> None:
    axis = np.linspace(-3.0, 3.0, 25)
    x, y = np.meshgrid(axis, axis)
    z = 0.08 * (x * x + y * y)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    expected = np.column_stack((-0.16 * x.ravel(), -0.16 * y.ravel(), np.ones(x.size)))
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    builder = SurfaceBuilder(0.35)
    normals = builder.estimate_normals(points, outward_reference=expected)
    curvature = builder.estimate_curvatures(points, normals)
    center = np.linalg.norm(points[:, :2], axis=1) < 2.0
    assert np.median(curvature[center, 0, 0]) < -0.1


def test_molecular_surface_is_deterministic_and_compact() -> None:
    atoms = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], dtype=np.float32)
    radii = np.array([1.7, 1.7], dtype=np.float32)
    builder = SurfaceBuilder(0.8, 1.4)
    first, _ = builder.build(atoms, radii)
    second, _ = builder.build(atoms, radii)
    assert 100 < len(first["surface_positions"]) < 1000
    for name in first:
        assert np.array_equal(first[name], second[name])
    assert first["surface_edge_index"].shape[0] == 2
    assert first["surface_area_weights"].dtype == np.float32
