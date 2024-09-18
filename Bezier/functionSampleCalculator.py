import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

# Definições do Usuario:
    # Dominio da função:
xMinValue = -5
xMaxValue = 5
yMinValue = -5
yMaxValue = 5

QtdDePontosDeControle = 3  # OBS: quantidade de em cada eixo

    # Definição da Função:
def funcao(x, y):
    return np.sin(x*y)


# Gerando valores igualmente espaçados para X e Y utilizando np.linspace()
X = np.linspace(xMinValue, xMaxValue, QtdDePontosDeControle)
Y = np.linspace(yMinValue, yMaxValue, QtdDePontosDeControle)

x, y = np.meshgrid(X, Y)

z = funcao(x, y)


print("\n x:\n")
print(x)
print("\n y:\n")
print(y)
print("\n z:\n")
print(z)


# Pontos de Controle:
x = np.array([[-0.5, -2, 0], [1, 1, 1], [2, 2, 2]])
y = np.array([[2, 1, 0], [2, 0, -1], [2, 1, 1]])
z = np.array([[1, -1, 2], [0, -0.5, 2], [0.5, 1, 2]])

print("\n x:\n")
print(x)
print("\n y:\n")
print(y)
print("\n z:\n")
print(z)
