import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Definições do Usuario:

    # Raio da Esfera:
r = 5

    # Dominio da função:
thetaMinValue = 0
thetaMaxValue = 2*np.pi
phiMinValue = 0
phiMaxValue = np.pi

QtdDePontosDeControle = 15  # OBS: quantidade de pontos de controle

# Generate evenly spaced values for x and y using np.linspace
theta = np.linspace(thetaMinValue, thetaMaxValue, QtdDePontosDeControle)
phi = np.linspace(phiMinValue, phiMaxValue, QtdDePontosDeControle)

theta, phi = np.meshgrid(theta, phi)

    # Definição das Funções:
def funcaoX(theta, phi):
        return r * np.sin(phi) * np.cos(theta)

def funcaoY(theta, phi):
        return r * np.sin(phi) * np.sin(theta)

def funcaoZ(theta, phi):
        return r * np.cos(phi)

X = funcaoX(theta, phi)
Y = funcaoY(theta, phi)
Z = funcaoZ(theta, phi)

print(X.shape)
print(Y.shape)
print(Z.shape)

# Flatten the arrays for triangulation (Delaunay works on 2D arrays)
x = X.flatten()
y = Y.flatten()
z = Z.flatten()

print(x.shape)
print(y.shape)
print(z.shape)

# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DA SUPERFÍCIE DESTACANDO OS PONTOS DE CONTROLE

fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.view_init(elev=30, azim=42)  # Elevation = 30 degrees, Azimuth = 60 degrees

ax.scatter(x, y, z)
plt.show()
# -------------------------------------------------------------------------------------------------


points = np.vstack((theta.flatten(), phi.flatten())).T  # 2D array of (x, y) points for triangulation

print(points)

# Perform Delaunay triangulation
tri = Delaunay(points)

print(tri.simplices)

# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DA FUNÇÃO DESTACANDO A REDE DE TRIANGULOS QUE DEFINE A SUPERFÍCIE

def triangulosParaLados(triangulos):
    lados = set([])
    for tri in triangulos:
        for k in range(3):
            l1, l2 = tri[k], tri[(k+1)%3]
            l1, l2 = min(l1,l2), max(l1,l2)
            lados.add((l1,l2))
    return np.array(list(lados), dtype=int)

lados = triangulosParaLados(tri.simplices)

fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.scatter(x, y, z)

for lado in lados:
    xi = x[lado[0]]
    yi = y[lado[0]]
    zi = z[lado[0]]
    xj = x[lado[1]]
    yj = y[lado[1]]
    zj = z[lado[1]]

    ax.plot([xi, xj], [yi, yj], [zi, zj])
plt.show()
# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DESTACANDO CADA TRIANGULO QUE DEFINE A SUPERFÍCIE COM CORES DIFERENTES

verticesTriangulos = []
print(tri.simplices.shape)

for tri in tri.simplices:
    verticesTriangulos.append([ (x[tri[0]], y[tri[0]], z[tri[0]]), 
                                (x[tri[1]], y[tri[1]], z[tri[1]]), 
                                (x[tri[2]], y[tri[2]], z[tri[2]]) ])

colors = ['red', 'green', 'blue']

fig = plt.figure(figsize=(18,24))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=42)

# Loop through each triangle and add it to the plot
for i, verts in enumerate(verticesTriangulos):
    tri = Poly3DCollection([verts], color=colors[(i%3)], alpha=0.5)
    ax.add_collection3d(tri)

plt.show()

# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DA SUPERFÍCIE SEM DESTAQUE NOS PONTOS DE CONTROLE E COM AS ARESTAS DOS TRIANGULOS FINAS

fig = plt.figure(figsize=(18,24))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=42)

for i, verts in enumerate(verticesTriangulos):
    tri = Poly3DCollection([verts], color="blue", alpha=0.5, edgecolor='none', linewidth=0)
    ax.add_collection3d(tri)

plt.show()
# -------------------------------------------------------------------------------------------------