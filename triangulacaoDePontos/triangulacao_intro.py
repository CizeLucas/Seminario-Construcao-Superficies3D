import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# Definições do Usuario:

    # Dominio da função:
xMinValue = -10
xMaxValue = 10
yMinValue = -10
yMaxValue = 10

QtdDePontosDeControle = 50


    # Definição da Função:
def funcao(x, y):
        return np.sin(x) * np.cos(y)


valores_X = np.linspace(xMinValue, xMaxValue, QtdDePontosDeControle)
valores_Y = np.linspace(yMinValue, yMaxValue, QtdDePontosDeControle)

x, y = np.meshgrid(valores_X, valores_Y)

z = funcao(x, y)
z_zeros = np.zeros(z.shape)

print(x.shape)
print(y.shape)
print(z.shape)

fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.scatter(x, y, z_zeros)
plt.show()

fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.scatter(x, y, z)
plt.show()