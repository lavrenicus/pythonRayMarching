"""Tests for ray marching utilities."""

import unittest
import numpy as np
from objects import Vector, Sphere, Plane
from utils import clamp, clamp_array, ray_march_vectorized, compute_normals_from_scene


class TestClamp(unittest.TestCase):
    def test_within_range(self):
        self.assertEqual(clamp(5, 0, 10), 5)

    def test_below_min(self):
        self.assertEqual(clamp(-1, 0, 10), 0)

    def test_above_max(self):
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_at_min(self):
        self.assertEqual(clamp(0, 0, 10), 0)

    def test_at_max(self):
        self.assertEqual(clamp(10, 0, 10), 10)


class TestClampArray(unittest.TestCase):
    def test_clamp_array(self):
        arr = np.array([-1, 0, 5, 10, 15])
        result = clamp_array(arr, 0, 10)
        np.testing.assert_array_equal(result, [0, 0, 5, 10, 10])

    def test_clamp_array_no_change(self):
        arr = np.array([2, 5, 8])
        result = clamp_array(arr, 0, 10)
        np.testing.assert_array_equal(result, [2, 5, 8])


class TestRayMarchVectorized(unittest.TestCase):
    def setUp(self):
        self.sphere = Sphere((0, 0, 5), 2)
        self.plane = Plane((0, -2, 0))
        self.scene = [self.sphere, self.plane]

    def test_hit_sphere_center(self):
        origin = np.array([0.0, 0.0, 0.0])
        dirs = np.array([[[0.0, 0.0, 1.0]]])
        d, n = ray_march_vectorized(origin, dirs, self.scene, max_distance=20)
        self.assertAlmostEqual(d[0, 0], 3.0, places=0)

    def test_hit_sphere_offset(self):
        origin = np.array([0.0, 0.0, 0.0])
        dirs = np.array([[[0.0, 0.0, 1.0]]])
        d, n = ray_march_vectorized(origin, dirs, [self.sphere], max_distance=20)
        self.assertGreater(d[0, 0], 0)
        self.assertLess(d[0, 0], 10)

    def test_miss_all(self):
        origin = np.array([0.0, 10.0, 0.0])
        dirs = np.array([[[0.0, 0.0, 1.0]]])
        d, n = ray_march_vectorized(origin, dirs, self.scene, max_distance=5)
        self.assertEqual(d[0, 0], 5.0)

    def test_output_shapes(self):
        origin = np.array([0.0, 0.0, 0.0])
        dirs = np.zeros((4, 6, 3), dtype=np.float64)
        dirs[:, :, 2] = 1.0
        d, n = ray_march_vectorized(origin, dirs, self.scene)
        self.assertEqual(d.shape, (4, 6))
        self.assertEqual(n.shape, (4, 6, 3))

    def test_normals_are_unit(self):
        origin = np.array([0.0, 0.0, 0.0])
        dirs = np.zeros((2, 2, 3), dtype=np.float64)
        dirs[:, :, 2] = 1.0
        d, n = ray_march_vectorized(origin, dirs, self.scene, max_distance=20)
        lengths = np.linalg.norm(n, axis=-1)
        hit_mask = d < 20
        np.testing.assert_allclose(lengths[hit_mask], 1.0, atol=1e-5)

    def test_distances_non_negative(self):
        origin = np.array([0.0, 0.0, 0.0])
        dirs = np.zeros((3, 3, 3), dtype=np.float64)
        dirs[:, :, 2] = 1.0
        d, _ = ray_march_vectorized(origin, dirs, self.scene)
        self.assertTrue(np.all(d >= 0))

    def test_distances_bounded(self):
        origin = np.array([0.0, 0.0, 0.0])
        dirs = np.zeros((3, 3, 3), dtype=np.float64)
        dirs[:, :, 2] = 1.0
        d, _ = ray_march_vectorized(origin, dirs, self.scene, max_distance=10)
        self.assertTrue(np.all(d <= 10))


class TestComputeNormalsFromScene(unittest.TestCase):
    def test_plane_normal(self):
        plane = Plane((0, -2, 0))
        pts = np.array([[[0, 0, 0], [5, 0, 0], [10, 0, 0]]])
        n = compute_normals_from_scene(pts, [plane])
        np.testing.assert_allclose(n[0, :, 1], 1.0, atol=0.1)
        np.testing.assert_allclose(n[0, :, 0], 0.0, atol=0.1)
        np.testing.assert_allclose(n[0, :, 2], 0.0, atol=0.1)

    def test_normals_are_unit(self):
        sphere = Sphere((0, 0, 5), 2)
        pts = np.array([[[0, 0, 3], [0, 0, 7]]])
        n = compute_normals_from_scene(pts, [sphere])
        lengths = np.linalg.norm(n, axis=-1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-4)
