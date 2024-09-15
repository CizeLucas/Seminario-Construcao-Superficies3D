import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# Step 1: Generate random points in 3D (x, y, z)
x = np.random.rand(50)
y = np.random.rand(50)

print(x.shape)
print(x)

z = np.sin(x) * np.cos(y)  # Some surface function

print(z.shape)

points = np.vstack((x, y)).T  # 2D array of (x, y) points for triangulation

# Step 2: Perform Delaunay triangulation in the xy-plane
tri = Delaunay(points)

# Step 3: Plot the triangulated surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use triangulation data to create a surface
ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap=plt.cm.Spectral)

plt.show()
