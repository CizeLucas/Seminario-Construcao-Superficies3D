import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

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
        return (math.factorial(n) / (math.factorial(i) * math.factorial(n - i)))

def Mj(m, j):
    return (math.factorial(m) / (math.factorial(j) * math.factorial(m - j)))

# Funções para calcular as Bases do Polinômio de Bernstein
def J(n, i, u):
    return np.matrix(Ni(n, i) * (u ** i) * (1 - u) ** (n - i))

def K(m, j, w):
    return np.matrix(Mj(m, j) * (w ** j) * (1 - w) ** (m - j))

# LOOPS PRINCIPAIS:
for i in range(0, uPTS):
    for j in range(0, wPTS):
        
        Jt = J(n, i, u)
        Kt = K(m, j, w)

        # Armazena os resultados das funções Basis
        b.append(Jt)
        d.append(Kt)

        # Calcula a transposta do array Jt
        Jt = Jt.transpose()

        # Calculando o ponto da curva Bezier da iteração
        xBezier = Jt * Kt * x[i, j] + xBezier
        yBezier = Jt * Kt * y[i, j] + yBezier
        zBezier = Jt * Kt * z[i, j] + zBezier


# Plotagem:

fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.view_init(elev=30, azim=42)  # Elevation = 30 degrees, Azimuth = 60 degrees
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

colors = ["red", "green", "blue"]
for i in range(3):
    ax.scatter(x[i], y[i], z[i], color=colors[i])

plt.show()


if(False):
    # Valores dos polinômios de Berstein
    plt.figure()
    plt.subplot(121)
    for line in b:
        plt.plot(u, line.transpose())
    plt.show()

    plt.figure()
    plt.subplot(121)
    for line in d:
        plt.plot(w, line.transpose())
    plt.show()

print(xBezier)
print(yBezier)
print(zBezier)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xBezier, yBezier, zBezier)
ax.scatter(x, y, z, edgecolors="face")
plt.show()