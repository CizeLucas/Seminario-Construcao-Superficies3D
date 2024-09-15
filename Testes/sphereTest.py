import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Definitions
r = 5
thetaMinValue = 0
thetaMaxValue = 2*np.pi
phiMinValue = 0
phiMaxValue = np.pi
QtdDePontosDeControle = 15  # Number of control points

# Generate evenly spaced values for theta and phi
theta = np.linspace(thetaMinValue, thetaMaxValue, QtdDePontosDeControle)
phi = np.linspace(phiMinValue, phiMaxValue, QtdDePontosDeControle)
theta, phi = np.meshgrid(theta, phi)

# Define functions
def funcaoX(theta, phi):
    return r * np.sin(phi) * np.cos(theta)

def funcaoY(theta, phi):
    return r * np.sin(phi) * np.sin(theta)

def funcaoZ(theta, phi):
    return r * np.cos(phi)

# Compute Cartesian coordinates
X = funcaoX(theta, phi)
Y = funcaoY(theta, phi)
Z = funcaoZ(theta, phi)

# Flatten arrays for triangulation
x = X.flatten()
y = Y.flatten()
z = Z.flatten()

# Convert spherical coordinates to 2D projection for triangulation
points = np.vstack((theta.flatten(), phi.flatten())).T

# Perform Delaunay triangulation on the 2D projection
tri = Delaunay(points)

# -------------------------------------------------------------------------------------------------
# Plot the surface with triangulation

fig = plt.figure(figsize=(18, 24))
ax = plt.axes(projection="3d")
ax.view_init(elev=30, azim=42)  # Elevation = 30 degrees, Azimuth = 60 degrees

# Plot the surface
ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', alpha=0.6)
plt.show()
# -------------------------------------------------------------------------------------------------
