import numpy as np
import pytest
from scipy.spatial import cKDTree

from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive


@pytest.mark.parametrize("resolution", [0.3, 1.0, 2.4])
def test_batched_burial_preserves_scalar_decisions_and_threshold_ties(resolution) -> None:
    """Sparse vectorization keeps exact historical survivors, including near-cutoff pairs."""
    rng = np.random.default_rng(67)
    atoms = rng.uniform(-5, 5, (75, 3))
    radii = rng.uniform(2.5, 3.5, len(atoms))
    points = rng.uniform(-8, 8, (4103, 3))
    owners = rng.integers(0, len(atoms), len(points))
    tolerance = max(1e-5, resolution * 0.02)
    boundary = radii[1] - tolerance
    distances = [np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)]
    points[:3] = atoms[1] + np.asarray([[distance, 0, 0] for distance in distances])
    owners[:3] = 0

    tree = cKDTree(atoms)
    keep = np.ones(len(points), bool)
    for index, nearby in enumerate(tree.query_ball_point(points, float(radii.max()) + .05 *
                                                        resolution)):
        for other in nearby:
            if other != owners[index] and float(np.linalg.norm(points[index] - atoms[other])) < (
                float(radii[other]) - tolerance
            ):
                keep[index] = False
                break
    actual, actual_owners = SurfaceBuilder(resolution)._remove_buried(points, owners, atoms, radii)
    assert np.array_equal(actual, points[keep])
    assert np.array_equal(actual_owners, owners[keep])


def test_envelope_validation_is_independent_of_model_top_k() -> None:
    """Larger, more distant spheres still contribute to the physical outward normal."""
    atoms = np.repeat([[0.0, 0.0, -2.8], [-3.2, 0.0, 0.0]], [32, 64], axis=0)
    radii = np.repeat([1.4, 1.8], [32, 64])
    normal = np.asarray([[64.0, 0.0, 32.0]])
    normal /= np.linalg.norm(normal)
    arrays = {
        "atom_positions": atoms,
        "vdw_radii": radii,
        "surface_positions": np.zeros((1, 3)),
        "surface_normals": normal,
    }
    gaps, cosines = ProteinArchive(PreprocessConfig())._envelope_alignment(arrays)
    assert gaps[0] == pytest.approx(0, abs=1e-12)
    assert cosines[0] == pytest.approx(1)
    assert normal[0, 2] < 0.99  # Top-32 centers alone incorrectly predict a vertical normal.


def test_surface_reference_resolves_real_1hty_active_sphere_rounding(monkeypatch) -> None:
    """One actual 1HTY_A point toggles a sphere across the hard cutoff after float32 storage."""
    atoms = np.asarray([
        [-22.2434139251709, -9.465559959411621, 17.605642318725586],
        [-23.007413864135742, -8.970560073852539, 18.635643005371094],
        [-17.130413055419922, -7.28656005859375, 17.020641326904297],
        [-22.788414001464844, -4.3835601806640625, 14.412642478942871],
        [-24.06141471862793, -6.555560111999512, 14.545641899108887],
        [-17.742414474487305, -4.897560119628906, 16.927642822265625],
        [-17.509414672851562, -8.634559631347656, 17.63064193725586],
        [-21.847414016723633, -4.706560134887695, 15.136642456054688],
        [-17.876413345336914, -6.264559745788574, 17.42264175415039],
        [-18.871414184570312, -4.0695600509643555, 17.548643112182617],
    ], dtype=np.float32)
    radii = np.asarray([1.55, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.52, 1.55, 1.7], np.float32)
    reference = np.asarray([[-20.98182753784089, -6.732233427707495, 17.053360392252603]])
    normal = np.asarray([[0.37713553530274174, -0.4874564127312504, 0.7874992277450786]])
    archive = ProteinArchive(PreprocessConfig(surface_resolution=1.0))
    arrays = {
        "atom_positions": atoms,
        "vdw_radii": radii,
        "surface_positions": reference.astype(np.float32),
        "surface_normals": normal.astype(np.float32),
    }
    _, rounded_cosines = archive._envelope_alignment(arrays)
    gaps, cosines = archive._envelope_alignment(arrays, reference)
    assert rounded_cosines[0] < 0.99
    assert cosines[0] > 0.999999
    assert abs(gaps[0]) < 0.025

    # Exercise the lazy base-sampler confirmation, not just the envelope helper. This exact
    # reproduced point is the scientific reference; unrelated points may never excuse a normal.
    arrays.update({
        "surface_curvatures": np.zeros((1, len(archive.config.curvature_scales), 3)),
        "surface_neighbors": np.zeros((1, 1), dtype=np.int32),
        "surface_neighbor_mask": np.zeros((1, 1), dtype=bool),
        "surface_neighbor_distances": np.zeros((1, 1)),
    })
    monkeypatch.setattr(SurfaceBuilder, "sample", lambda *_: (reference, np.asarray([7])))
    assert archive.surface_diagnostics(arrays)["minimum_normal_cosine"] > 0.999999

    arrays["surface_normals"] *= -1
    assert archive._envelope_alignment(arrays, reference)[1][0] < -0.999999
    with pytest.raises(ValueError, match="outward molecular envelope"):
        archive.surface_diagnostics(arrays)
    with pytest.raises(ValueError, match="rounding precision"):
        archive._envelope_alignment(arrays, reference + 0.01)


def test_envelope_batches_preserve_all_points_and_reject_unsupported_positions() -> None:
    """The final partial batch and points without any nearby sphere keep explicit diagnostics."""
    arrays = {
        "atom_positions": np.zeros((1, 3)),
        "vdw_radii": np.asarray([1.7]),
        "surface_positions": np.tile([3.1, 0.0, 0.0], (2051, 1)),
        "surface_normals": np.tile([1.0, 0.0, 0.0], (2051, 1)),
    }
    arrays["surface_positions"][-1] = [100, 100, 100]
    gaps, cosines = ProteinArchive(PreprocessConfig())._envelope_alignment(arrays)
    assert np.allclose(gaps[:-1], 0)
    assert np.allclose(cosines[:-1], 1)
    assert np.isposinf(gaps[-1])
    assert cosines[-1] == 0


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
    assert "surface_edge_index" not in first
    assert first["surface_area_weights"].dtype == np.float32
