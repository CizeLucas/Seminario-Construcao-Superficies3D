import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
from utilities import *

# Pontos de Controle:
x = np.array([[-0.5, -2, 0], [1, 1, 1], [2, 2, 2]])
y = np.array([[2, 1, 0], [2, 0, -1], [2, 1, 1]])
z = np.array([[1, -1, 2], [0, -0.5, 2], [0.5, 1, 2]])

# Numeros de células em cada direção:
uCells = 12
wCells = 10

# VARIAVEIS DEPENDENTES:
# Numero total de Pontos de Controle na direção U e W:
uPTS = np.size(x, 0)
wPTS = np.size(x, 1)

# Numero total de subdivisões:
n = uPTS - 1
m = wPTS - 1

# Definição do vetor de variação das variáveis paramétricas:
u = np.linspace(0, 1, uCells)
w = np.linspace(0, 1, wCells)

# Bases do polinomio de Bernstein
b = []
d = []

# Inicializando as matrizes vazias para armazenar X, Y e Z das curvas de Bezier
xBezier = np.zeros((uCells, wCells))
yBezier = np.zeros((uCells, wCells))
zBezier = np.zeros((uCells, wCells))

# Funções para calcular os coeficientes binomiais:
def Ni(n, i):
        return (np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i)))

def Mj(m, j):
    return (np.math.factorial(m) / (np.math.factorial(j) * np.math.factorial(m - j)))

# Funções para calcular as Bases do Polinômio de Bernstein


print(x)
print(y)
print(z)

print(u)
print(w)


def calculate_value(x, y):
    value = x**2 + y**2 - 5
    if(value < 0):
        value = value * -1
    result = - math.sqrt(value)
    return result

X = np.arange(0, 5, 0.1)
Y = np.arange(0, 5, 0.1)
Z = np.zeros((X.size, Y.size))

for i in range(X.size):
    Z[i] = calculate_value(X[i], Y[i])

# Plotando pontos unitários em 3D
fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.scatter(X, Y, Z)
plt.show()