"""Vectorized ray marching engine using numpy for parallel CPU execution.

All pixels are processed simultaneously via numpy array operations,
replacing the original per-pixel Python loop for significant speedup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from objects import Transform


def ray_march_vectorized(
    origin: np.ndarray,
    directions: np.ndarray,
    scene: list[Transform],
    max_steps: int = 100,
    max_distance: float = 25.0,
    min_distance: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_normals_from_scene(
    points: np.ndarray,
    scene: list[Transform],
    epsilon: float = 0.0005,
) -> np.ndarray:
    """Compute normals by sampling SDF at offset points.

    Uses central differences: normal = normalize(SDF(p+e) - SDF(p-e)).

    Args:
        points: (H, W, 3) positions.
        scene: list of scene objects.
        epsilon: finite difference step size.

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
