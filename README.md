# Python Ray Marching

A real-time 3D renderer using ray marching with numpy vectorization for parallel CPU execution. Renders scenes defined with signed distance functions (SDFs) and displays results with pygame.

![Rendered scene with spheres and plane](screenshot.png)

## Features

- **Vectorized rendering** — all pixels processed simultaneously via numpy arrays
- **Signed distance functions** — spheres, planes, and extensible primitives
- **Diffuse lighting** — per-pixel shading based on surface normals
- **Real-time display** — live rendering window with pygame
- **Image export** — saves rendered output as PNG

## Requirements

- Python 3.10+
- pygame >= 2.5.0
- Pillow >= 10.0.0
- numpy >= 1.24.0

## Installation

```bash
git clone https://github.com/yourusername/pythonRayMarching.git
cd pythonRayMarching
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This will:
1. Render the default scene (two spheres on a plane)
2. Display the result in a pygame window
3. Save the rendered image to a temporary PNG file
4. Open the image with your system's default viewer

Press **Escape** or close the window to exit.

## Examples

### Basic Scene

```python
import objects
import utils
from main import render, display_with_pygame

# Create objects
sphere = objects.Sphere(position=(0, 0, 5), radius=2)
camera = objects.Camera(position=(0, 0, 0))
light = objects.Transform(position=(1, 1, -1))

# Render
rgb = render([sphere], camera, light, width=640, height=480)
display_with_pygame(rgb)
```

### Multiple Spheres

```python
scene = [
    objects.Sphere(position=(-2, 0, 5), radius=2),
    objects.Sphere(position=(2, 0, 5), radius=2),
    objects.Sphere(position=(0, 2, 5), radius=1),
]
camera = objects.Camera(position=(0, 0, 0))
light = objects.Transform(position=(1, 1, -1))

rgb = render(scene, camera, light)
```

### Adding a Ground Plane

```python
scene = [
    objects.Sphere(position=(0, 1, 5), radius=1),
    objects.Plane(position=(0, -1, 0)),  # ground at y = -1
]
camera = objects.Camera(position=(0, 2, 0))
light = objects.Transform(position=(1, 1, -1))

rgb = render(scene, camera, light, width=800, height=600)
```

### Custom Camera and Light

```python
camera = objects.Camera(position=(5, 3, -2))  # side view
light = objects.Transform(position=(-1, 2, 1))  # light from above-left

rgb = render(scene, camera, light)
```

## How It Works

Ray marching traces a ray from the camera through each pixel and iteratively steps along the ray until it hits a surface.

### Algorithm

1. **Generate rays** — for each pixel, compute a ray direction from the camera through the pixel
2. **March** — step along the ray by the distance returned by the scene's SDF (Signed Distance Function)
3. **Hit test** — if the step distance is below a threshold, we've hit a surface
4. **Shade** — compute the surface normal and apply diffuse lighting

### Signed Distance Functions

Each object defines an SDF that returns the closest distance from any point to its surface:

- **Sphere**: `distance = length(point - center) - radius`
- **Plane**: `distance = point.y - plane.y`

The SDF property ensures we can safely step forward by exactly that distance without overshooting any surface.

### Vectorization

Instead of a Python `for` loop over each pixel, numpy processes all pixels simultaneously:

```python
# All ray directions as a (H, W, 3) array
directions = np.stack([uv_x, uv_y, np.ones_like(uv_x)], axis=-1)

# March all rays at once
for step in range(max_steps):
    points = origin + directions * distances[..., np.newaxis]
    min_dist = np.min([obj.sdf_array(points) for obj in scene], axis=0)
    distances += min_dist
```

## Project Structure

```
pythonRayMarching/
├── main.py              # Entry point, scene setup, rendering pipeline
├── objects/
│   └── __init__.py      # Vector math, Sphere, Plane, Camera classes
├── utils/
│   └── __init__.py      # Vectorized ray marching engine
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## Extending

### Add a New Primitive

```python
class Torus(objects.Transform):
    def __init__(self, position, major_radius, minor_radius):
        super().__init__(position)
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def get_distance_to_surface(self, point):
        # Torus SDF implementation
        ...

    def sdf_array(self, points):
        # Vectorized torus SDF
        ...
```

### Change Render Resolution

```python
rgb = render(scene, camera, light, width=1920, height=1080)
```

## License

MIT
