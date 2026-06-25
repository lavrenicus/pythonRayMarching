"""Tests for new primitives: Torus, Cylinder, Box, Replica."""

import unittest
import numpy as np
from objects import Vector, Sphere, Torus, Cylinder, Box, Replica


class TestTorus(unittest.TestCase):
    def test_init(self):
        t = Torus((1, 2, 3), major_radius=5, minor_radius=2)
        self.assertEqual(t.position, [1, 2, 3])
        self.assertEqual(t.major_radius, 5)
        self.assertEqual(t.minor_radius, 2)

    def test_distance_at_center(self):
        t = Torus((0, 0, 0), 5, 2)
        d = t.get_distance_to_surface(Vector((0, 0, 0)))
        self.assertAlmostEqual(d, 3.0)

    def test_distance_on_major_ring(self):
        t = Torus((0, 0, 0), 5, 2)
        d = t.get_distance_to_surface(Vector((5, 0, 0)))
        self.assertAlmostEqual(d, -2.0)

    def test_distance_outside(self):
        t = Torus((0, 0, 0), 5, 2)
        d = t.get_distance_to_surface(Vector((10, 0, 0)))
        self.assertAlmostEqual(d, 3.0)

    def test_sdf_array(self):
        t = Torus((0, 0, 0), 5, 2)
        pts = np.array([[[5, 0, 0], [0, 0, 0], [10, 0, 0]]])
        result = t.sdf_array(pts)
        np.testing.assert_array_almost_equal(result[0], [-2.0, 3.0, 3.0])

    def test_sdf_matches_single(self):
        t = Torus((1, 2, 3), 5, 2)
        pt = Vector((6, 2, 3))
        single = t.get_distance_to_surface(pt)
        arr = t.sdf_array(np.array([[[6, 2, 3]]]))
        self.assertAlmostEqual(float(arr[0, 0]), single)


class TestCylinder(unittest.TestCase):
    def test_init(self):
        c = Cylinder((1, 2, 3), radius=5)
        self.assertEqual(c.position, [1, 2, 3])
        self.assertEqual(c.radius, 5)

    def test_distance_inside(self):
        c = Cylinder((0, 0, 0), 3)
        d = c.get_distance_to_surface(Vector((0, 0, 0)))
        self.assertAlmostEqual(d, -3.0)

    def test_distance_on_surface(self):
        c = Cylinder((0, 0, 0), 3)
        d = c.get_distance_to_surface(Vector((3, 5, 0)))
        self.assertAlmostEqual(d, 0.0)

    def test_distance_outside(self):
        c = Cylinder((0, 0, 0), 3)
        d = c.get_distance_to_surface(Vector((5, 10, 0)))
        self.assertAlmostEqual(d, 2.0)

    def test_infinite_along_y(self):
        c = Cylinder((0, 0, 0), 3)
        d1 = c.get_distance_to_surface(Vector((0, 0, 0)))
        d2 = c.get_distance_to_surface(Vector((0, 100, 0)))
        self.assertAlmostEqual(d1, d2)

    def test_sdf_array(self):
        c = Cylinder((0, 0, 0), 3)
        pts = np.array([[[3, 5, 0], [0, 0, 0], [5, 10, 0]]])
        result = c.sdf_array(pts)
        np.testing.assert_array_almost_equal(result[0], [0.0, -3.0, 2.0])


class TestBox(unittest.TestCase):
    def test_init(self):
        b = Box((1, 2, 3), half_extents=(4, 5, 6))
        self.assertEqual(b.position, [1, 2, 3])
        self.assertEqual(b.half_extents, (4, 5, 6))

    def test_distance_inside(self):
        b = Box((0, 0, 0), (1, 1, 1))
        d = b.get_distance_to_surface(Vector((0, 0, 0)))
        self.assertAlmostEqual(d, -1.0)

    def test_distance_on_surface(self):
        b = Box((0, 0, 0), (1, 1, 1))
        d = b.get_distance_to_surface(Vector((1, 1, 1)))
        self.assertAlmostEqual(d, 0.0)

    def test_distance_outside(self):
        b = Box((0, 0, 0), (1, 1, 1))
        d = b.get_distance_to_surface(Vector((2, 0, 0)))
        self.assertAlmostEqual(d, 1.0)

    def test_distance_corner(self):
        b = Box((0, 0, 0), (1, 1, 1))
        d = b.get_distance_to_surface(Vector((2, 2, 2)))
        self.assertAlmostEqual(d, np.sqrt(3), places=5)

    def test_normal_outside(self):
        b = Box((0, 0, 0), (1, 1, 1))
        n = b.get_normal(Vector((2, 0, 0)))
        self.assertAlmostEqual(n.x, 1.0)
        self.assertAlmostEqual(n.y, 0.0)
        self.assertAlmostEqual(n.z, 0.0)

    def test_sdf_array(self):
        b = Box((0, 0, 0), (1, 1, 1))
        pts = np.array([[[0, 0, 0], [1, 1, 1], [2, 0, 0]]])
        result = b.sdf_array(pts)
        np.testing.assert_array_almost_equal(result[0], [-1.0, 0.0, 1.0])

    def test_sdf_matches_single(self):
        b = Box((1, 2, 3), (2, 3, 4))
        pt = Vector((4, 5, 7))
        single = b.get_distance_to_surface(pt)
        arr = b.sdf_array(np.array([[[4, 5, 7]]]))
        self.assertAlmostEqual(float(arr[0, 0]), single)


class TestReplica(unittest.TestCase):
    def test_init(self):
        s = Sphere((0, 0, 0), 1)
        r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        self.assertIs(r.child, s)
        self.assertEqual(r.cell_size, (4, 4, 4))

    def test_distance_at_cell_center(self):
        s = Sphere((0, 0, 0), 1)
        r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        d = r.get_distance_to_surface(Vector((2, 2, 2)))
        self.assertAlmostEqual(d, -1.0)

    def test_distance_at_cell_origin(self):
        s = Sphere((0, 0, 0), 1)
        r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        d = r.get_distance_to_surface(Vector((0, 0, 0)))
        self.assertAlmostEqual(d, 2.464102, places=4)

    def test_distance_repeated(self):
        s = Sphere((0, 0, 0), 1)
        r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        d1 = r.get_distance_to_surface(Vector((2, 2, 2)))
        d2 = r.get_distance_to_surface(Vector((6, 6, 6)))
        self.assertAlmostEqual(d1, d2)

    def test_distance_between(self):
        s = Sphere((0, 0, 0), 1)
        r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        d = r.get_distance_to_surface(Vector((3, 1.5, 1.5)))
        self.assertGreater(d, 0)

    def test_sdf_array(self):
        s = Sphere((0, 0, 0), 1)
        r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        pts = np.array([[[2, 2, 2], [6, 6, 6], [10, 10, 10]]])
        result = r.sdf_array(pts)
        np.testing.assert_array_almost_equal(result[0], [-1.0, -1.0, -1.0])

    def test_with_torus(self):
        t = Torus((0, 0, 0), 3, 1)
        r = Replica((0, 0, 0), child=t, cell_size=(10, 10, 10))
        d = r.get_distance_to_surface(Vector((8, 5, 5)))
        self.assertAlmostEqual(d, -1.0)
