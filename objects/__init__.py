"""Vector math and 3D scene primitives for ray marching.

Provides Vector for point/direction math, and scene objects
(Sphere, Plane, Camera) with both single-point and vectorized SDF methods.
"""

from math import sqrt
import numpy as np


class BadParametersException(Exception):
    """Raised when function arguments are invalid."""


class Vector(list):
    """3D vector backed by a Python list.

    Supports basic arithmetic (+, -, *, /), dot product,
    normalization, distance calculation, and x/y/z accessors.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def __sub__(self, vec):
        if isinstance(vec, (float, int)):
            vec = self._scalar_to_vector(vec)
        if len(self) != len(vec):
            raise BadParametersException('Vector dimensions mismatch')
        return Vector([self[i] - vec[i] for i in range(len(self))])

    def __add__(self, vec):
        if isinstance(vec, (float, int)):
            vec = self._scalar_to_vector(vec)
        return Vector([self[i] + vec[i] for i in range(len(self))])

    def __mul__(self, n):
        if isinstance(n, (float, int)):
            return Vector([v * n for v in self])
        if isinstance(n, list):
            return Vector([self[i] * n[i] for i in range(len(self))])
        return NotImplemented

    def __truediv__(self, n):
        return Vector([v / n for v in self])

    def __repr__(self) -> str:
        return 'Vector:({})'.format(', '.join(str(v) for v in self))

    def _scalar_to_vector(self, value):
        return Vector([value] * len(self))

    def dot(self, vec):
        """Return dot product of two vectors.

        >>> Vector((1, 3, -5)).dot(Vector((4, -2, -1)))
        3
        """
        return sum(self[i] * vec[i] for i in range(len(self)))

    def normalized(self):
        """Return unit vector in the same direction.

        >>> Vector((0, 2, 0)).normalized()
        Vector:(0.0, 1.0, 0.0)
        >>> Vector((156, 0, 0)).normalized()
        Vector:(1.0, 0.0, 0.0)
        >>> Vector((0, 0, 5432)).normalized()
        Vector:(0.0, 0.0, 1.0)
        """
        length_val = self.length
        inv_length = 1.0 / length_val
        return Vector([v * inv_length for v in self])

    def distance(self, vec):
        """Return Euclidean distance to another vector."""
        return sqrt(sum((self[i] - vec[i]) ** 2 for i in range(len(self))))

    @property
    def length(self):
        """Return scalar length of vector.

        >>> Vector((3, 4, 0)).length
        5.0
        """
        return sqrt(sum(v * v for v in self))

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @x.setter
    def x(self, value):
        self[0] = value

    @y.setter
    def y(self, value):
        self[1] = value

    @z.setter
    def z(self, value):
        self[2] = value

    def to_numpy(self):
        """Convert to numpy array of shape (3,)."""
        return np.array([self[0], self[1], self[2]])


class Transform:
    """Base class for scene objects with a position.

    Subclasses must implement get_distance_to_surface() and get_normal().
    """

    _position = None

    def __init__(self, position):
        if len(position) != 3:
            raise BadParametersException('Position must be a 3D vector')
        self._position = Vector(position)

    def get_distance_to_surface(self, point):
        """Return signed distance from point to surface. Override in subclass."""
        raise NotImplementedError()

    def get_normal(self, point):
        """Return surface normal at point. Override in subclass."""
        raise NotImplementedError()

    def sdf_array(self, points):
        """Vectorized SDF for numpy array of shape (..., 3).

        Default implementation calls per-point SDF via vectorize.
        Subclasses should override for better performance.
        """
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        result = np.array([self.get_distance_to_surface(Vector(p)) for p in flat])
        return result.reshape(original_shape)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = Vector(value)


class Plane(Transform):
    """Infinite horizontal plane at y = position.y."""

    def get_distance_to_surface(self, point):
        return point.y - self.position.y

    def get_normal(self, point):
        return Vector((0, 1, 0))

    def sdf_array(self, points):
        return points[..., 1] - self.position.y


class Sphere(Transform):
    """Sphere with center at position and given radius."""

    def __init__(self, position, radius):
        super().__init__(position)
        self.radius = radius

    def get_distance_to_surface(self, point):
        return self.position.distance(point) - self.radius

    def get_normal(self, point):
        return (self.position - point).normalized()

    def sdf_array(self, points):
        center = self.position.to_numpy()
        diff = points - center
        return np.linalg.norm(diff, axis=-1) - self.radius


class Camera(Transform):
    """Camera positioned in the scene."""
