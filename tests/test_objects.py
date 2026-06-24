"""Tests for scene objects (Sphere, Plane, Camera, Transform)."""

import unittest
import numpy as np
from objects import Vector, Transform, Sphere, Plane, Camera, BadParametersException


class TestTransform(unittest.TestCase):
    def test_init(self):
        t = Transform((1, 2, 3))
        self.assertEqual(t.position, [1, 2, 3])

    def test_init_invalid(self):
        with self.assertRaises(BadParametersException):
            Transform((1, 2))

    def test_position_setter(self):
        t = Transform((1, 2, 3))
        t.position = (4, 5, 6)
        self.assertEqual(t.position, [4, 5, 6])

    def test_get_distance_not_implemented(self):
        t = Transform((0, 0, 0))
        with self.assertRaises(NotImplementedError):
            t.get_distance_to_surface(Vector((0, 0, 0)))

    def test_get_normal_not_implemented(self):
        t = Transform((0, 0, 0))
        with self.assertRaises(NotImplementedError):
            t.get_normal(Vector((0, 0, 0)))


class TestSphere(unittest.TestCase):
    def test_init(self):
        s = Sphere((1, 2, 3), 5)
        self.assertEqual(s.position, [1, 2, 3])
        self.assertEqual(s.radius, 5)

    def test_distance_at_center(self):
        s = Sphere((0, 0, 0), 5)
        d = s.get_distance_to_surface(Vector((0, 0, 0)))
        self.assertAlmostEqual(d, -5.0)

    def test_distance_on_surface(self):
        s = Sphere((0, 0, 0), 5)
        d = s.get_distance_to_surface(Vector((5, 0, 0)))
        self.assertAlmostEqual(d, 0.0)

    def test_distance_outside(self):
        s = Sphere((0, 0, 0), 5)
        d = s.get_distance_to_surface(Vector((0, 0, 10)))
        self.assertAlmostEqual(d, 5.0)

    def test_distance_offset_center(self):
        s = Sphere((0, 0, 5), 3)
        d = s.get_distance_to_surface(Vector((0, 0, 5)))
        self.assertAlmostEqual(d, -3.0)

    def test_normal_at_surface(self):
        s = Sphere((0, 0, 0), 5)
        n = s.get_normal(Vector((5, 0, 0)))
        self.assertAlmostEqual(n.x, -1.0)
        self.assertAlmostEqual(n.y, 0.0)
        self.assertAlmostEqual(n.z, 0.0)

    def test_normal_is_unit(self):
        s = Sphere((1, 2, 3), 5)
        n = s.get_normal(Vector((1, 2, 8)))
        self.assertAlmostEqual(n.length, 1.0)

    def test_sdf_array(self):
        s = Sphere((0, 0, 0), 5)
        pts = np.array([[[0, 0, 0], [0, 0, 10]]])
        result = s.sdf_array(pts)
        np.testing.assert_array_almost_equal(result, [[-5.0, 5.0]])

    def test_sdf_array_matches_single(self):
        s = Sphere((2, 3, 4), 3)
        pt = Vector((5, 3, 4))
        single = s.get_distance_to_surface(pt)
        arr = s.sdf_array(np.array([[[5, 3, 4]]]))
        self.assertAlmostEqual(float(arr[0, 0]), single)


class TestPlane(unittest.TestCase):
    def test_distance_above(self):
        p = Plane((0, -2, 0))
        d = p.get_distance_to_surface(Vector((0, 0, 0)))
        self.assertEqual(d, 2)

    def test_distance_on_plane(self):
        p = Plane((0, -2, 0))
        d = p.get_distance_to_surface(Vector((0, -2, 0)))
        self.assertEqual(d, 0)

    def test_distance_below(self):
        p = Plane((0, -2, 0))
        d = p.get_distance_to_surface(Vector((0, -5, 0)))
        self.assertEqual(d, -3)

    def test_normal_always_up(self):
        p = Plane((0, -2, 0))
        n = p.get_normal(Vector((100, 50, 200)))
        self.assertEqual(n, [0, 1, 0])

    def test_sdf_array(self):
        p = Plane((0, -2, 0))
        pts = np.array([[[0, 0, 0], [0, -2, 0], [0, -5, 0]]])
        result = p.sdf_array(pts)
        np.testing.assert_array_almost_equal(result, [[2, 0, -3]])

    def test_sdf_array_matches_single(self):
        p = Plane((0, -3, 0))
        pt = Vector((0, 0, 0))
        single = p.get_distance_to_surface(pt)
        arr = p.sdf_array(np.array([[[0, 0, 0]]]))
        self.assertAlmostEqual(float(arr[0, 0]), single)


class TestCamera(unittest.TestCase):
    def test_init(self):
        c = Camera((0, 1, -10))
        self.assertEqual(c.position, [0, 1, -10])

    def test_is_transform(self):
        c = Camera((0, 0, 0))
        self.assertIsInstance(c, Transform)
