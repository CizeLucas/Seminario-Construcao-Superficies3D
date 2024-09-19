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

QtdDePontosDeControle = 7  # OBS: quantidade de em cada eixo

    # Definição da Função:
def funcao(x, y):
    return np.sin(x) * np.cos(y) # Superfície de ondulação de sen() e cos()

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
uCells = 50
wCells = 50

# VARIAVEIS DEPENDENTES:
# Numero total de Pontos de Controle na direção U e W:
uPTS = np.size(x, 0) # definidos pela quantidade de pontos de controle
wPTS = np.size(x, 1)

if(False):
    print(uPTS)
    print(wPTS)

# Numero total de subdivisões (regiões entre os pontos de controle):
n = uPTS - 1 # n e m definirão o grau da curva
m = wPTS - 1

# Definição do vetor de variação das variáveis paramétricas:
u = np.linspace(0, 1, uCells)
w = np.linspace(0, 1, wCells)

if(False):
    print(u)
    print(w)

# Vetores para armazenar Bases do polinomio de Bernstein
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

# Funções para calcular o Polinômio de Bernstein
def J(n, i, u): # direção u
    return np.matrix(Ni(n, i) * (u ** i) * (1 - u) ** (n - i))

def K(m, j, w): # direção w
    return np.matrix(Mj(m, j) * (w ** j) * (1 - w) ** (m - j))

# LOOPS PRINCIPAIS:
for i in range(0, uPTS): # Primeiro Somatório -> adiciona a dimensionalidade Y
    for j in range(0, wPTS): # Segundo Somatório -> calcula os pontos no eixo X e Z
        
        
        Jt = J(n, i, u)
        Kt = K(m, j, w)

        # Armazena os resultados das funções Basis
        b.append(Jt)
        d.append(Kt)

        # Calcula a transposta do array Jt
        Jt = Jt.T

        # Calculando o ponto da curva Bezier da iteração
        xBezier = Jt * Kt * x[i, j] + xBezier
        yBezier = Jt * Kt * y[i, j] + yBezier
        zBezier = Jt * Kt * z[i, j] + zBezier
#    break # Faz com que o somatório mais interno da equação da superfície de Bézier só seja calculado uma vez.


# Plotagem dos pontos de cotrole:
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
    
    # Gráfico com os valores dos polinômios de Berstein
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

if(False):
    print(xBezier)
    print(yBezier)
    print(zBezier)
    print(zBezier.shape)


# Plota a superfície no gráfico COM os pontos de controle
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xBezier, yBezier, zBezier)

# Desenhar os outros pontos
for i in range(QtdDePontosDeControle):
    for j in range(QtdDePontosDeControle):
        ax.scatter(x[i, j], y[i, j], z[i, j], color="blue")

# Destacar os primeiros e últimos pontos de controle
ax.scatter(x[:, 0], y[:, 0], z[:, 0], color='black', s=100) # Destaca os pontos de controle do INÍCIO da curvas de bézier 
ax.scatter(x[:, -1], y[:, -1], z[:, -1], color='black', s=100) # Destaca os pontos de controle do FIM da curvas de bézier 

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# Plota a superfície no gráfico SEM os pontos de controle
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xBezier, yBezier, zBezier)
plt.show()