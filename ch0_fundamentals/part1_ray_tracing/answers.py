# [0.1] Ray Tracing - https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/
#
# Paste the page's setup code into the first cell, then work through the exercises cell by cell.
# Run a cell with the "Run Cell" link above each `# %%`, or select lines and press Shift+Enter.

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import jaxtyping
except:
    %pip install jaxtyping einops

# `go.FigureWidget` (used in the interactive plots below) needs anywidget, which plotly doesn't
# install for you
try:
    import anywidget
except:
    %pip install anywidget

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"
# %%
# 1D Image Rendering
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros(num_pixels, 2, 3)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays


tests.test_make_rays_1d(make_rays_1d)

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
# %%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
# %%
# Ray-Object Intersection
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    a = segment[0] - segment[1]
    b = (segment[0] - ray[0])[0:2]
    matrix = t.stack([ray[1], a], dim=1)[0:2]

    assert matrix.shape == (2,2)
    assert b.shape == (2,)

    if t.linalg.matrix_rank(matrix) != 2:
        return False
    
    solution = t.linalg.solve(matrix, b)
    
    if solution[0] >= 0 and 0 <= solution[1] <= 1:
        return True
    else:
        return False


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
# Batched Ray-Segment Intersection
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    origins = rays[:, 0, :2]
    directions = rays[:, 1, :2]
    L1 = segments[:, 0, :2]
    L2 = segments[:, 1, :2]

    nrays = len(rays)
    nsegments = len(segments)

    mats = t.zeros(nrays, nsegments, 2, 2)
    mats[..., 0] = directions[:, None, :]
    mats[..., 1] = (L1 - L2)[None, ...]

    determinants = t.linalg.det(mats)
    is_singular = determinants.abs() < 1E-6

    mats[is_singular, ...] = t.eye(2)
    vecs = L1[None, :, :] - origins[:, None, :]

    solution = t.linalg.solve(mats, vecs)
    u, v = solution[..., 0], solution[..., 1]

    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
# 2D Rays
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    rays = t.zeros(num_pixels_y, 2, 3)  
    rays[..., 1, 0] = 1
    t.linspace(-y_limit, y_limit, num_pixels_y, out=rays[:, 1, 1])
    rays = einops.repeat(rays, "y p d -> (y z) p d", z=num_pixels_z)
    rays[:, 1, 2] = t.linspace(-z_limit, z_limit, num_pixels_z).repeat(num_pixels_y)
    return rays


tests.test_make_rays_2d(make_rays_2d)

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%
# Triangle Coordinates
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    mat = t.stack([-D, B - A, C - A], dim=-1)
    vec = O - A
    if t.linalg.det(mat).abs() < 1E-6:
        return False
    s, u, v = t.linalg.solve(mat, vec)
    if (s >= 0) & (u >= 0) & (v >= 0) & (u+v <= 1):
        return True
    else:
        return False


tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%
# Single-Triangle Rendering
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    BA = einops.repeat(triangle[1] - triangle[0], "b -> nrays b", nrays = len(rays))
    CA = einops.repeat(triangle[2] - triangle[0], "b -> nrays b", nrays = len(rays))
    mat = t.stack([-rays[:, 1, :], BA, CA], dim=-1)
    vec = rays[:, 0, :] - triangle[0]

    is_singular = t.linalg.det(mat).abs() < 1E-6
    mat[is_singular] = t.eye(3)
    sol = t.linalg.solve(mat, vec)

    return (sol[:, 0] >= 0) & (sol[:, 1] >= 0) & (sol[:, 2] >= 0) & (sol[:, 1] + sol[:, 2] <= 1) & (~is_singular)


tests.test_raytrace_triangle(raytrace_triangle)

A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
# %%
# Mesh Rendering
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    NR = len(rays)
    NT = len(triangles)
    R = einops.repeat(rays, "r p d -> r t p d", t=NT)
    T = einops.repeat(triangles, "t p d -> r t p d", r=NR)

    mats = t.stack([-R[..., 1, :], T[..., 1, :] - T[..., 0, :], T[..., 2, :] - T[..., 0, :]], dim=-1)
    assert mats.shape == (NR, NT, 3, 3)

    vecs = R[..., 0, :] - T[..., 0, :]
    assert vecs.shape == (NR, NT, 3)

    is_singular = t.linalg.det(mats).abs() < 1E-6
    mats[is_singular] = t.eye(3)

    sol = t.linalg.solve(mats, vecs)
    s, u, v = sol.unbind(dim=-1)
    intersects = (s >= 0) & (u >= 0) & (v >= 0) & (u+v <= 1) & (~is_singular)
    s[~intersects] = float('inf')
    s[intersects] = s[intersects] * R[intersects, 1, 0]

    return einops.reduce(s, "r t -> r", "min")


tests.test_raytrace_mesh(raytrace_mesh)

num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
# %%
