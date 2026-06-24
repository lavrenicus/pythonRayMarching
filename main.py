"""Ray marching renderer using numpy for parallel CPU execution.

Renders a 3D scene with spheres using signed distance functions,
displays with pygame, and saves the output as an image.
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pygame
from pygame import gfxdraw
from PIL import Image

import objects
import utils


WIDTH = 640
HEIGHT = 480


def create_scene():
    """Create the default scene with two spheres.

    Returns:
        scene: list of scene objects.
        camera: Camera object.
        light: Transform representing light direction.
    """
    sphere1 = objects.Sphere(position=(-4, 0, 10), radius=4)
    sphere2 = objects.Sphere(position=(4, 0, 10), radius=4)
    plane = objects.Plane(position=(0, -4, 0))
    camera = objects.Camera(position=(0, 1, -10))
    light = objects.Transform(position=(1, 1, 1))

    scene = [sphere1, sphere2, plane]
    return scene, camera, light


def render(scene, camera, light, width=WIDTH, height=HEIGHT,
           max_steps=100, max_distance=50.0):
    """Render the scene using vectorized ray marching.

    All pixels are processed simultaneously via numpy arrays.

    Args:
        scene: list of scene objects.
        camera: Camera object.
        light: Transform with light direction.
        width: render width in pixels.
        height: render height in pixels.
        max_steps: max ray marching steps.
        max_distance: max ray travel distance.

    Returns:
        rgb: numpy array of shape (height, width, 3), dtype uint8.
    """
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    uv_x = (xx - width / 2.0) / height
    uv_y = -(yy - height / 2.0) / height

    directions = np.stack([uv_x, uv_y, np.ones_like(uv_x)], axis=-1)
    lengths = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / lengths

    origin = camera.position.to_numpy()
    light_dir = light.position.normalized().to_numpy()
    light_dir = light_dir / np.linalg.norm(light_dir)

    start = time.time()
    distances, normals = utils.ray_march_vectorized(
        origin, directions, scene,
        max_steps=max_steps,
        max_distance=max_distance,
        min_distance=0.001
    )
    elapsed = time.time() - start
    print(f"Ray march: {elapsed:.3f}s ({width}x{height})")

    diffuse = np.einsum('ijk,k->ij', normals, light_dir)
    diffuse = np.clip(diffuse, 0.0, 1.0)

    background = np.array([0.15, 0.15, 0.25])
    sky_mask = distances >= max_distance
    diffuse[sky_mask] = 0.0

    rgb = (diffuse[..., np.newaxis] * np.array([200, 210, 230])).astype(np.float64)
    rgb[sky_mask] = background * 255

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def display_with_pygame(rgb, width=WIDTH, height=HEIGHT):
    """Display the rendered image in a pygame window.

    Args:
        rgb: (H, W, 3) numpy array of uint8 RGB values.
        width: window width.
        height: window height.
    """
    pygame.init()
    surf = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ray Marching - numpy vectorized")

    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    surf.blit(surface, (0, 0))
    pygame.display.update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
    pygame.quit()


def save_image(rgb, filepath):
    """Save RGB array as an image file.

    Args:
        rgb: (H, W, 3) numpy array.
        filepath: output file path.
    """
    img = Image.fromarray(rgb, 'RGB')
    img.save(filepath)
    print(f"Saved: {filepath}")


def open_image(filepath):
    """Open image with system default viewer.

    Args:
        filepath: path to the image file.
    """
    if sys.platform == 'win32':
        os.startfile(filepath)
    elif sys.platform == 'darwin':
        subprocess.run(['open', filepath])
    else:
        subprocess.run(['xdg-open', filepath])


def main():
    """Render scene, display with pygame, save output image."""
    scene, camera, light = create_scene()

    rgb = render(scene, camera, light, width=WIDTH, height=HEIGHT)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        filepath = f.name
    save_image(rgb, filepath)

    display_with_pygame(rgb)
    open_image(filepath)


if __name__ == '__main__':
    main()
