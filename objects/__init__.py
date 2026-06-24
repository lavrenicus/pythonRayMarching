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

    >>> v = Vector((1, 2, 3))
    >>> v.x, v.y, v.z
    (1, 2, 3)
    >>> repr(v)
    'Vector:(1, 2, 3)'
    """

    def __init__(self, *args):
        """Create a vector from a sequence of numbers.

        >>> Vector((1, 2, 3))
        Vector:(1, 2, 3)
        >>> Vector([4, 5, 6])
        Vector:(4, 5, 6)
        """
        super().__init__(*args)

    def __sub__(self, vec):
        """Subtract another vector or scalar.

        >>> Vector((1, 2, 3)) - Vector((4, 5, 6))
        Vector:(-3, -3, -3)
        >>> Vector((1, 2, 3)) - 1
        Vector:(0, 1, 2)
        """
        if isinstance(vec, (float, int)):
            vec = self._scalar_to_vector(vec)
        if len(self) != len(vec):
            raise BadParametersException('Vector dimensions mismatch')
        return Vector([self[i] - vec[i] for i in range(len(self))])

    def __add__(self, vec):
        """Add another vector or scalar.

        >>> Vector((1, 2, 3)) + Vector((4, 5, 6))
        Vector:(5, 7, 9)
        >>> Vector((1, 2, 3)) + 10
        Vector:(11, 12, 13)
        """
        if isinstance(vec, (float, int)):
            vec = self._scalar_to_vector(vec)
        return Vector([self[i] + vec[i] for i in range(len(self))])

    def __mul__(self, n):
        """Multiply by a scalar or element-wise by another vector.

        >>> Vector((1, 2, 3)) * 2
        Vector:(2, 4, 6)
        >>> Vector((1, 2, 3)) * Vector((2, 3, 4))
        Vector:(2, 6, 12)
        """
        if isinstance(n, (float, int)):
            return Vector([v * n for v in self])
        if isinstance(n, list):
            return Vector([self[i] * n[i] for i in range(len(self))])
        return NotImplemented

    def __truediv__(self, n):
        """Divide by a scalar.

        >>> Vector((2, 4, 6)) / 2
        Vector:(1.0, 2.0, 3.0)
        """
        return Vector([v / n for v in self])

    def __repr__(self) -> str:
        """Return string representation.

        >>> repr(Vector((1, 2, 3)))
        'Vector:(1, 2, 3)'
        """
        return 'Vector:({})'.format(', '.join(str(v) for v in self))

    def _scalar_to_vector(self, value):
        """Broadcast scalar to vector of same length.

        >>> Vector((1, 2, 3))._scalar_to_vector(5)
        Vector:(5, 5, 5)
        """
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
        """Return Euclidean distance to another vector.

        >>> Vector((0, 0, 0)).distance(Vector((3, 4, 0)))
        5.0
        """
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
        """X component.

        >>> Vector((1, 2, 3)).x
        1
        """
        return self[0]

    @property
    def y(self):
        """Y component.

        >>> Vector((1, 2, 3)).y
        2
        """
        return self[1]

    @property
    def z(self):
        """Z component.

        >>> Vector((1, 2, 3)).z
        3
        """
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
        """Convert to numpy array of shape (3,).

        >>> Vector((1, 2, 3)).to_numpy()
        array([1, 2, 3])
        """
        return np.array([self[0], self[1], self[2]])


class Transform:
    """Base class for scene objects with a position.

    Subclasses must implement get_distance_to_surface() and get_normal().
    """

    _position = None

    def __init__(self, position):
        """Initialize with a 3D position.

        >>> t = Transform((1, 2, 3))
        >>> t.position
        Vector:(1, 2, 3)
        """
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
        """Object position as a Vector.

        >>> Transform((1, 2, 3)).position
        Vector:(1, 2, 3)
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = Vector(value)


class Plane(Transform):
    """Infinite horizontal plane at y = position.y.

    >>> p = Plane((0, -2, 0))
    >>> p.get_distance_to_surface(Vector((0, 0, 0)))
    2
    >>> p.get_distance_to_surface(Vector((0, -2, 0)))
    0
    """

    def get_distance_to_surface(self, point):
        """Return signed distance from point to plane.

        >>> Plane((0, -2, 0)).get_distance_to_surface(Vector((0, 0, 0)))
        2
        >>> Plane((0, -2, 0)).get_distance_to_surface(Vector((0, -5, 0)))
        -3
        """
        return point.y - self.position.y

    def get_normal(self, point):
        """Return plane normal (always (0, 1, 0)).

        >>> Plane((0, 0, 0)).get_normal(Vector((1, 2, 3)))
        Vector:(0, 1, 0)
        """
        return Vector((0, 1, 0))

    def sdf_array(self, points):
        """Vectorized SDF for numpy array.

        >>> import numpy as np
        >>> p = Plane((0, -2, 0))
        >>> pts = np.array([[[0, 0, 0], [0, -2, 0], [0, -5, 0]]])
    >>> p.sdf_array(pts)
    array([[ 2,  0, -3]])
        """
        return points[..., 1] - self.position.y


class Sphere(Transform):
    """Sphere with center at position and given radius.

    >>> s = Sphere((0, 0, 0), 5)
    >>> s.radius
    5
    """

    def __init__(self, position, radius):
        """Create sphere with center and radius.

        >>> s = Sphere((1, 2, 3), 5)
        >>> s.position
        Vector:(1, 2, 3)
        >>> s.radius
        5
        """
        super().__init__(position)
        self.radius = radius

    def get_distance_to_surface(self, point):
        """Return signed distance from point to sphere surface.

        >>> Sphere((0, 0, 0), 5).get_distance_to_surface(Vector((0, 0, 0)))
        -5.0
        >>> Sphere((0, 0, 0), 5).get_distance_to_surface(Vector((0, 0, 10)))
        5.0
        """
        return self.position.distance(point) - self.radius

    def get_normal(self, point):
        """Return unit normal pointing away from sphere center.

    >>> Sphere((0, 0, 0), 5).get_normal(Vector((5, 0, 0)))
    Vector:(-1.0, 0.0, 0.0)
        """
        return (self.position - point).normalized()

    def sdf_array(self, points):
        """Vectorized SDF for numpy array.

        >>> import numpy as np
        >>> s = Sphere((0, 0, 0), 5)
        >>> pts = np.array([[[0, 0, 0], [0, 0, 10]]])
    >>> s.sdf_array(pts)
    array([[-5.,  5.]])
        """
        center = self.position.to_numpy()
        diff = points - center
        return np.linalg.norm(diff, axis=-1) - self.radius


class Camera(Transform):
    """Camera positioned in the scene.

    >>> c = Camera((0, 1, -10))
    >>> c.position
    Vector:(0, 1, -10)
    """
