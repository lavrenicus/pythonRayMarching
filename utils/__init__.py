"""Vectorized ray marching engine using numpy for parallel CPU execution.

All pixels are processed simultaneously via numpy array operations,
replacing the original per-pixel Python loop for significant speedup.
"""

import numpy as np


def get_distance_vectorized(points, scene):
    """Compute SDF distances and normals for all points at once.

    Args:
        points: numpy array of shape (H, W, 3) - all sample points.
        scene: list of Transform objects with sdf_array() methods.

    Returns:
        distances: (H, W) array of minimum distances to scene.
        normals: (H, W, 3) array of surface normals.
    """
    all_distances = np.stack(
        [obj.sdf_array(points) for obj in scene], axis=-1
    )  # shape: (H, W, num_objects)

    min_indices = np.argmin(all_distances, axis=-1)  # (H, W)
    distances = np.min(all_distances, axis=-1)  # (H, W)

    normals = compute_normals_vectorized(points, distances)
    return distances, normals


def compute_normals_vectorized(points, distances, epsilon=0.001):
    """Compute surface normals via central differences on the distance field.

    Args:
        points: (H, W, 3) sample positions.
        distances: (H, W) distance values.
        epsilon: finite difference step size.

    Returns:
        normals: (H, W, 3) unit normals.
    """
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dz = np.zeros_like(points)

    dx[:, 1:, 0] = distances[:, 1:] - distances[:, :-1]
    dx[:, :-1, 0] += distances[:, 1:] - distances[:, :-1]

    dy[1:, :, 1] = distances[1:, :] - distances[:-1, :]
    dy[:-1, :, 1] += distances[1:, :] - distances[:-1, :]

    dz[:, :, 2] = distances  # z-component approximation

    grad = np.stack([
        (np.roll(distances, -1, axis=1) - np.roll(distances, 1, axis=1)),
        (np.roll(distances, -1, axis=0) - np.roll(distances, 1, axis=0)),
        np.zeros_like(distances)
    ], axis=-1)

    lengths = np.linalg.norm(grad, axis=-1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    return grad / lengths


def ray_march_vectorized(origin, directions, scene, max_steps=100,
                          max_distance=25.0, min_distance=0.001):
    """Ray march all rays simultaneously using numpy.

    Args:
        origin: (3,) array - camera position.
        directions: (H, W, 3) array - ray directions (normalized).
        scene: list of scene objects.
        max_steps: maximum marching steps.
        max_distance: maximum ray travel distance.
        min_distance: surface hit threshold.

    Returns:
        distances: (H, W) array of total travel distances.
        normals: (H, W, 3) array of surface normals at hit points.
    """
    h, w = directions.shape[:2]
    distances = np.zeros((h, w), dtype=np.float64)
    hit_mask = np.zeros((h, w), dtype=bool)

    for _step in range(max_steps):
        points = origin + directions * distances[..., np.newaxis]

        scene_dists = np.stack(
            [obj.sdf_array(points) for obj in scene], axis=-1
        )  # (H, W, num_objects)
        min_dist = np.min(scene_dists, axis=-1)  # (H, W)

        new_hit = (~hit_mask) & (min_dist < min_distance)
        past_max = (~hit_mask) & (distances >= max_distance)
        done = new_hit | past_max

        distances += min_dist * (~hit_mask).astype(np.float64)
        distances = np.minimum(distances, max_distance)
        hit_mask |= done

        if np.all(hit_mask):
            break

    distances = np.minimum(distances, max_distance)
    points = origin + directions * distances[..., np.newaxis]
    normals = compute_normals_from_scene(points, scene)
    return distances, normals


def compute_normals_from_scene(points, scene, epsilon=0.0005):
    """Compute normals by sampling SDF at offset points.

    Uses central differences: normal = normalize(SDF(p+e) - SDF(p-e)).

    Args:
        points: (H, W, 3) positions.
        scene: list of scene objects.

    Returns:
        (H, W, 3) unit normal vectors.
    """
    e_x = np.array([epsilon, 0.0, 0.0])
    e_y = np.array([0.0, epsilon, 0.0])
    e_z = np.array([0.0, 0.0, epsilon])

    def sdf_at(offsets):
        """Compute minimum SDF distance across all scene objects for given points."""
        d = np.stack(
            [obj.sdf_array(offsets) for obj in scene], axis=-1
        )
        return np.min(d, axis=-1)

    grad_x = sdf_at(points + e_x) - sdf_at(points - e_x)
    grad_y = sdf_at(points + e_y) - sdf_at(points - e_y)
    grad_z = sdf_at(points + e_z) - sdf_at(points - e_z)

    grad = np.stack([grad_x, grad_y, grad_z], axis=-1)
    lengths = np.linalg.norm(grad, axis=-1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    return grad / lengths


def clamp(x, a, b):
    """Clamp x to range [a, b].

    >>> clamp(5, 0, 3)
    3
    >>> clamp(-1, 0, 1)
    0
    """
    return max(a, min(b, x))


def clamp_array(arr, a, b):
    """Clamp numpy array to range [a, b]."""
    return np.clip(arr, a, b)
