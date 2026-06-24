"""Tests for Vector class."""

import unittest
import numpy as np
from objects import Vector, BadParametersException


class TestVectorCreation(unittest.TestCase):
    def test_from_tuple(self):
        v = Vector((1, 2, 3))
        self.assertEqual(v, [1, 2, 3])

    def test_from_list(self):
        v = Vector([4, 5, 6])
        self.assertEqual(v, [4, 5, 6])

    def test_repr(self):
        v = Vector((1, 2, 3))
        self.assertEqual(repr(v), 'Vector:(1, 2, 3)')


class TestVectorArithmetic(unittest.TestCase):
    def test_sub_vectors(self):
        result = Vector((1, 2, 3)) - Vector((4, 5, 6))
        self.assertEqual(result, [-3, -3, -3])

    def test_sub_scalar(self):
        result = Vector((1, 2, 3)) - 1
        self.assertEqual(result, [0, 1, 2])

    def test_add_vectors(self):
        result = Vector((1, 2, 3)) + Vector((4, 5, 6))
        self.assertEqual(result, [5, 7, 9])

    def test_add_scalar(self):
        result = Vector((1, 2, 3)) + 10
        self.assertEqual(result, [11, 12, 13])

    def test_mul_scalar(self):
        result = Vector((1, 2, 3)) * 2
        self.assertEqual(result, [2, 4, 6])

    def test_mul_vector(self):
        result = Vector((1, 2, 3)) * Vector((2, 3, 4))
        self.assertEqual(result, [2, 6, 12])

    def test_mul_not_implemented(self):
        result = Vector((1, 2, 3)).__mul__("bad")
        self.assertEqual(result, NotImplemented)

    def test_truediv(self):
        result = Vector((2, 4, 6)) / 2
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_dimension_mismatch(self):
        with self.assertRaises(BadParametersException):
            Vector((1, 2)) - Vector((1, 2, 3))


class TestVectorProperties(unittest.TestCase):
    def test_x(self):
        self.assertEqual(Vector((1, 2, 3)).x, 1)

    def test_y(self):
        self.assertEqual(Vector((1, 2, 3)).y, 2)

    def test_z(self):
        self.assertEqual(Vector((1, 2, 3)).z, 3)

    def test_set_x(self):
        v = Vector((1, 2, 3))
        v.x = 10
        self.assertEqual(v.x, 10)

    def test_set_y(self):
        v = Vector((1, 2, 3))
        v.y = 10
        self.assertEqual(v.y, 10)

    def test_set_z(self):
        v = Vector((1, 2, 3))
        v.z = 10
        self.assertEqual(v.z, 10)

    def test_length(self):
        self.assertAlmostEqual(Vector((3, 4, 0)).length, 5.0)

    def test_length_3d(self):
        self.assertAlmostEqual(Vector((1, 2, 2)).length, 3.0)


class TestVectorOperations(unittest.TestCase):
    def test_dot(self):
        self.assertEqual(Vector((1, 3, -5)).dot(Vector((4, -2, -1))), 3)

    def test_dot_orthogonal(self):
        self.assertEqual(Vector((1, 0, 0)).dot(Vector((0, 1, 0))), 0)

    def test_normalized(self):
        n = Vector((0, 2, 0)).normalized()
        self.assertAlmostEqual(n.x, 0.0)
        self.assertAlmostEqual(n.y, 1.0)
        self.assertAlmostEqual(n.z, 0.0)

    def test_normalized_length_one(self):
        n = Vector((3, 4, 5)).normalized()
        self.assertAlmostEqual(n.length, 1.0)

    def test_distance(self):
        d = Vector((0, 0, 0)).distance(Vector((3, 4, 0)))
        self.assertAlmostEqual(d, 5.0)

    def test_distance_same_point(self):
        d = Vector((1, 2, 3)).distance(Vector((1, 2, 3)))
        self.assertAlmostEqual(d, 0.0)

    def test_to_numpy(self):
        arr = Vector((1, 2, 3)).to_numpy()
        np.testing.assert_array_equal(arr, np.array([1, 2, 3]))

    def test_scalar_to_vector(self):
        result = Vector((1, 2, 3))._scalar_to_vector(5)
        self.assertEqual(result, [5, 5, 5])
