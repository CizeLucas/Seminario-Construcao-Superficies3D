import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

# Definições do Usuario:

    # Dominio da função:
xMinValue = 0
xMaxValue = 2*np.pi
yMinValue = 0
yMaxValue = 2*np.pi

QtdDePontosDeControle = 10  # OBS: quantidade de em cada eixo

    # Definição da Função:
def funcao(x, y):
    return np.sin(x) * np.cos(y)

# Gerando valores igualmente espaçados para X e Y utilizando np.linspace()
X = np.linspace(xMinValue, xMaxValue, QtdDePontosDeControle)
Y = np.linspace(yMinValue, yMaxValue, QtdDePontosDeControle)

# Pontos de controle X, Y e Z para a superfície de Bezier
x, y = np.meshgrid(X, Y)
z = funcao(x, y)

if(False):
    print("\n x:\n")
    print(x)
    print("\n y:\n")
    print(y)
    print("\n z:\n")
    print(z)

# Numeros de células em cada direção:
uCells = 40
wCells = 40

# VARIAVEIS DEPENDENTES:
# Numero total de Pontos de Controle na direção U e W:
uPTS = np.size(x, 0)
wPTS = np.size(x, 1)

print(uPTS)
print(wPTS)

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
ax.view_init(elev=30, azim=42)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

colors = ["red", "green", "blue"]
for i in range(QtdDePontosDeControle):
    ax.scatter(x[i], y[i], z[i], color=colors[i%3])

plt.show()


if(True):
    cores = ["red", "green", "black", "blue"]
    
    # Valores dos polinômios de Berstein
    plt.figure()

    plt.subplot(121)
    plt.title('U')
    contador=0
    for line in b:
        plt.plot(u, line.transpose(), color=cores[contador%4])
        contador+=1

    plt.subplot(122)
    plt.title('W')
    contador=0
    for line in d:
        plt.plot(w, line.transpose(), color=cores[contador%4])
        contador+=1

    plt.show()

print(xBezier)
print(yBezier)
print(zBezier)
print(zBezier.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xBezier, yBezier, zBezier)
ax.scatter(x, y, z, edgecolors="face")
plt.show()