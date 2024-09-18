import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import comb

# Bernstein polynomial function
def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * (1 - t) ** (n - i)

# Bézier surface generation function
def bezier_surface(control_points, u, v):
    n, m, _ = control_points.shape
    surface = np.zeros((len(u), len(v), 3))
    for i in range(n):
        for j in range(m):
            surface += np.outer(bernstein_poly(i, n-1, u), bernstein_poly(j, m-1, v)).reshape(len(u), len(v), 1) * control_points[i, j]
    return surface

# Control points for two Bézier patches
control_points_patch1 = np.array([
    [[0, 0, 0], [1, 0, 2], [2, 0, 0]],
    [[0, 1, 2], [1, 1, 3], [2, 1, 2]],
    [[0, 2, 0], [1, 2, 2], [2, 2, 0]]
])

control_points_patch2 = np.array([
    [[2, 0, 0], [3, 0, 2], [4, 0, 0]],
    [[2, 1, 2], [3, 1, 3], [4, 1, 2]],
    [[2, 2, 0], [3, 2, 2], [4, 2, 0]]
])

# Generate parameter values
u = np.linspace(0, 1, 30)
v = np.linspace(0, 1, 30)

# Generate the surface for each patch
surface_patch1 = bezier_surface(control_points_patch1, u, v)
surface_patch2 = bezier_surface(control_points_patch2, u, v)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot patch 1
ax.plot_surface(surface_patch1[:, :, 0], surface_patch1[:, :, 1], surface_patch1[:, :, 2], color='blue', alpha=0.6)

# Plot patch 2
ax.plot_surface(surface_patch2[:, :, 0], surface_patch2[:, :, 1], surface_patch2[:, :, 2], color='green', alpha=0.6)

# Set plot parameters
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
