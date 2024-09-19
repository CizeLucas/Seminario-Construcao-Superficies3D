import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

# Definições do Usuario:

    # Dominio da função:
xMinValue = 0
xMaxValue = 2*np.pi
yMinValue = 0
yMaxValue = 2*np.pi

QtdDePontosDeControle = 20  # OBS: quantidade de em cada eixo

seletorDeFuncao = 1 # Seleção da função a ser plotada (confira também o intervalo de domínio acima)

    # Definição da Função:

def funcao(x, y, select):
    match select:
        case 1:
            return (np.exp(x)*(np.sin(x) + np.cos(y))) # exp(Z) com Z sendo um número complexo
        case 2:
            return np.sqrt(x**2 + y**2) + 3*np.cos(np.sqrt(x**2 + y**2)) + 5 # Superfície Ondulada
        case 3:
            return x**2 + y**2 # Paraboloide Circular
        case 4:
            return np.sin(x*y)
        case _:
            return np.sin(x) * np.cos(y) # Superfície de ondulação de sen() e cos()


# Gera valores igualmente espaçados para a variação paramétrica de x e y
X = np.linspace(xMinValue, xMaxValue, QtdDePontosDeControle)
Y = np.linspace(yMinValue, yMaxValue, QtdDePontosDeControle)

if(True):
    print(X)
    print(Y)

# Gera as matrize X e Y para descrever o conjunto de pontos (x, y)
X, Y = np.meshgrid(X, Y)

if(True):
    print(X)
    print(Y)

# Calcula os valores de Z com X e Y
Z = funcao(X, Y, seletorDeFuncao)

if(True):
    print(Z)

# Flatten the arrays for triangulation (Delaunay works on 2D arrays)
x = X.flatten()
y = Y.flatten()
z = Z.flatten()
Z_zeros = np.zeros(z.shape) # gera um array de zeros do mesmo formato de z

# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DESTACANDO APENAS OS PONTOS DE IMAGEM UTILIZADOS

fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.view_init(elev=30, azim=42)

ax.scatter(x, y, Z_zeros)
plt.show()
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DA SUPERFÍCIE DESTACANDO OS PONTOS DE CONTROLE

fig = plt.figure(figsize=(18,24))
ax = plt.axes(projection="3d")
ax.view_init(elev=30, azim=42)

ax.scatter(x, y, z)
plt.show()
# -------------------------------------------------------------------------------------------------

# Gera um array com os pares de pontos (x, y) para serem triangulados
points = np.vstack((x, y)).T

# Calcula a triangulação com o Algoritimo de Delaunay (biblioteca Scipy)
tri = Delaunay(points)

# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DA FUNÇÃO DESTACANDO A REDE DE TRIANGULOS QUE DEFINE A SUPERFÍCIE

# Obtem os índices dos pares de aresta dos triangulos (exemplo: triangulo [79, 95, 110] -> [79, 95], [95, 110], [79, 110])
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

    # Plota as linhas arestas dos triangulos
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

# Itera sobre cada triangulo e aplica uma cor a sua face
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