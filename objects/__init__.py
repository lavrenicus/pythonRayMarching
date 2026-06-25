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


class Torus(Transform):
    """Torus (donut) with major radius R and minor radius r.

    Centered at position, lying in the XZ plane.

    >>> t = Torus((0, 0, 0), major_radius=5, minor_radius=2)
    >>> t.major_radius
    5
    >>> t.minor_radius
    2
    """

    def __init__(self, position, major_radius, minor_radius):
        """Create torus with center, major and minor radii.

        >>> t = Torus((1, 2, 3), 5, 2)
        >>> t.position
        Vector:(1, 2, 3)
        """
        super().__init__(position)
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def get_distance_to_surface(self, point):
        """Return signed distance from point to torus surface.

        >>> Torus((0, 0, 0), 5, 2).get_distance_to_surface(Vector((5, 0, 0)))
        -2.0
        >>> Torus((0, 0, 0), 5, 2).get_distance_to_surface(Vector((0, 0, 0)))
        3.0
        """
        p = point - self.position
        q = Vector((sqrt(p.x ** 2 + p.z ** 2) - self.major_radius, p.y, 0))
        return q.length - self.minor_radius

    def get_normal(self, point):
        """Return unit normal at point on torus surface.

        >>> n = Torus((0, 0, 0), 5, 2).get_normal(Vector((7, 0, 0)))
        >>> abs(abs(n.x) - 1.0) < 0.01
        True
        """
        p = point - self.position
        q_len = sqrt(p.x ** 2 + p.z ** 2)
        if q_len < 1e-8:
            return Vector((0, 1, 0))
        qx = (q_len - self.major_radius) * p.x / q_len
        qy = p.y
        return Vector((qx, qy, 0)).normalized()

    def sdf_array(self, points):
        """Vectorized SDF for numpy array.

        >>> import numpy as np
        >>> t = Torus((0, 0, 0), 5, 2)
        >>> pts = np.array([[[5, 0, 0], [0, 0, 0]]])
        >>> t.sdf_array(pts)
        array([[-2.,  3.]])
        """
        center = self.position.to_numpy()
        p = points - center
        xz_len = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
        qx = xz_len - self.major_radius
        qy = p[..., 1]
        return np.sqrt(qx ** 2 + qy ** 2) - self.minor_radius


class Cylinder(Transform):
    """Infinite cylinder along Y axis with given radius.

    Centered at position.

    >>> c = Cylinder((0, 0, 0), radius=3)
    >>> c.radius
    3
    """

    def __init__(self, position, radius):
        """Create cylinder with center and radius.

        >>> c = Cylinder((1, 2, 3), 5)
        >>> c.position
        Vector:(1, 2, 3)
        """
        super().__init__(position)
        self.radius = radius

    def get_distance_to_surface(self, point):
        """Return signed distance from point to cylinder surface.

        >>> Cylinder((0, 0, 0), 3).get_distance_to_surface(Vector((3, 5, 0)))
        0.0
        >>> Cylinder((0, 0, 0), 3).get_distance_to_surface(Vector((0, 0, 0)))
        -3.0
        >>> Cylinder((0, 0, 0), 3).get_distance_to_surface(Vector((5, 10, 0)))
        2.0
        """
        p = point - self.position
        d = sqrt(p.x ** 2 + p.z ** 2) - self.radius
        return d

    def get_normal(self, point):
        """Return unit normal at point.

        >>> n = Cylinder((0, 0, 0), 3).get_normal(Vector((3, 5, 0)))
        >>> abs(abs(n.x) - 1.0) < 0.01
        True
        """
        p = point - self.position
        d = sqrt(p.x ** 2 + p.z ** 2)
        if d < 1e-8:
            return Vector((1, 0, 0))
        return Vector((p.x / d, 0, p.z / d))

    def sdf_array(self, points):
        """Vectorized SDF for numpy array.

        >>> import numpy as np
        >>> c = Cylinder((0, 0, 0), 3)
        >>> pts = np.array([[[3, 5, 0], [0, 0, 0], [5, 10, 0]]])
        >>> c.sdf_array(pts)
        array([[ 0., -3.,  2.]])
        """
        center = self.position.to_numpy()
        p = points - center
        return np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2) - self.radius


class Box(Transform):
    """Axis-aligned box with half-extents (hx, hy, hz).

    Centered at position.

    >>> b = Box((0, 0, 0), half_extents=(2, 3, 4))
    >>> b.half_extents
    (2, 3, 4)
    """

    def __init__(self, position, half_extents):
        """Create box with center and half-extents.

        >>> b = Box((1, 2, 3), (4, 5, 6))
        >>> b.position
        Vector:(1, 2, 3)
        """
        super().__init__(position)
        self.half_extents = tuple(half_extents)

    def get_distance_to_surface(self, point):
        """Return signed distance from point to box surface.

        >>> Box((0, 0, 0), (1, 1, 1)).get_distance_to_surface(Vector((0, 0, 0)))
        -1.0
        >>> Box((0, 0, 0), (1, 1, 1)).get_distance_to_surface(Vector((1, 1, 1)))
        0.0
        >>> Box((0, 0, 0), (1, 1, 1)).get_distance_to_surface(Vector((2, 0, 0)))
        1.0
        """
        p = point - self.position
        hx, hy, hz = self.half_extents
        qx = max(abs(p.x) - hx, 0)
        qy = max(abs(p.y) - hy, 0)
        qz = max(abs(p.z) - hz, 0)
        outside = sqrt(qx * qx + qy * qy + qz * qz)
        inside = min(max(abs(p.x) - hx, max(abs(p.y) - hy, abs(p.z) - hz)), 0)
        return outside + inside

    def get_normal(self, point):
        """Return unit normal at point.

        >>> n = Box((0, 0, 0), (1, 1, 1)).get_normal(Vector((2, 0, 0)))
        >>> n
        Vector:(1.0, 0.0, 0.0)
        """
        p = point - self.position
        hx, hy, hz = self.half_extents
        nx = 1.0 if p.x > hx else (-1.0 if p.x < -hx else 0.0)
        ny = 1.0 if p.y > hy else (-1.0 if p.y < -hy else 0.0)
        nz = 1.0 if p.z > hz else (-1.0 if p.z < -hz else 0.0)
        n = Vector((nx, ny, nz))
        if n.length < 1e-8:
            return Vector((0, 1, 0))
        return n.normalized()

    def sdf_array(self, points):
        """Vectorized SDF for numpy array.

        >>> import numpy as np
        >>> b = Box((0, 0, 0), (1, 1, 1))
        >>> pts = np.array([[[0, 0, 0], [1, 1, 1], [2, 0, 0]]])
        >>> b.sdf_array(pts)
        array([[-1.,  0.,  1.]])
        """
        center = self.position.to_numpy()
        p = np.abs(points - center)
        hx, hy, hz = self.half_extents
        outside = np.sqrt(
            np.maximum(p[..., 0] - hx, 0) ** 2 +
            np.maximum(p[..., 1] - hy, 0) ** 2 +
            np.maximum(p[..., 2] - hz, 0) ** 2
        )
        inside = np.minimum(
            np.maximum(np.maximum(p[..., 0] - hx, p[..., 1] - hy), p[..., 2] - hz),
            0
        )
        return outside + inside


class Replica(Transform):
    """Repeats a child object in a grid pattern.

    Wraps any Transform object and tiles it at regular intervals.

    >>> s = Sphere((0, 0, 0), 1)
    >>> r = Replica((0, 0, 0), child=s, cell_size=(3, 3, 3))
    >>> r.cell_size
    (3, 3, 3)
    """

    def __init__(self, position, child, cell_size):
        """Create replica with child object and cell size.

        >>> s = Sphere((0, 0, 0), 1)
        >>> r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        >>> r.child is s
        True
        """
        super().__init__(position)
        self.child = child
        self.cell_size = tuple(cell_size)

    def _repeat(self, point):
        """Map point into base cell using modulo.

        >>> r = Replica((0, 0, 0), child=Sphere((0,0,0),1), cell_size=(4, 4, 4))
        >>> p = r._repeat(Vector((2, 2, 2)))
        >>> abs(p.x) < 0.01
        True
        """
        cx, cy, cz = self.cell_size
        p = point - self.position
        return Vector((
            (p.x % cx + cx) % cx - cx / 2,
            (p.y % cy + cy) % cy - cy / 2,
            (p.z % cz + cz) % cz - cz / 2,
        ))

    def get_distance_to_surface(self, point):
        """Return signed distance via repeated child SDF.

        >>> s = Sphere((0, 0, 0), 1)
        >>> r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        >>> d = r.get_distance_to_surface(Vector((2, 2, 2)))
        >>> abs(d - (-1.0)) < 0.01
        True
        """
        local = self._repeat(point)
        return self.child.get_distance_to_surface(local)

    def get_normal(self, point):
        """Return normal from repeated child.

        >>> s = Sphere((0, 0, 0), 1)
        >>> r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        >>> n = r.get_normal(Vector((3, 2, 2)))
        >>> abs(abs(n.x) - 1.0) < 0.01
        True
        """
        local = self._repeat(point)
        return self.child.get_normal(local)

    def sdf_array(self, points):
        """Vectorized SDF for numpy array.

        >>> import numpy as np
        >>> s = Sphere((0, 0, 0), 1)
        >>> r = Replica((0, 0, 0), child=s, cell_size=(4, 4, 4))
        >>> pts = np.array([[[2, 2, 2], [6, 6, 6]]])
        >>> r.sdf_array(pts)
        array([[-1., -1.]])
        """
        cx, cy, cz = self.cell_size
        center = self.position.to_numpy()
        p = points - center
        local = np.stack([
            (p[..., 0] % cx + cx) % cx - cx / 2,
            (p[..., 1] % cy + cy) % cy - cy / 2,
            (p[..., 2] % cz + cz) % cz - cz / 2,
        ], axis=-1)
        return self.child.sdf_array(local)
