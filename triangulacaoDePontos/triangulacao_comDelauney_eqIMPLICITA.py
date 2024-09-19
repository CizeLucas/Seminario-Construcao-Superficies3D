import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Definições do Usuario:

    # Intervalo paramétrico:
thetaMinValue = 0
thetaMaxValue = 2*np.pi
phiMinValue = 0
phiMaxValue = np.pi

    # Raio da Esfera:
r = 5

QtdDePontosDeControle = 15  # OBS: quantidade de pontos de controle

# Gera valores igualmente espaçados para a variação paramétrica de theta e phi
theta = np.linspace(thetaMinValue, thetaMaxValue, QtdDePontosDeControle)
phi = np.linspace(phiMinValue, phiMaxValue, QtdDePontosDeControle)


# Gera as matrize X e Y para descrever o conjunto de pontos de parametrização theta e phi
theta, phi = np.meshgrid(theta, phi)

    # Definição das Funções Parametrizadas da ESFERA:
def funcaoX(theta, phi):
        return r * np.sin(phi) * np.cos(theta)

def funcaoY(theta, phi):
        return r * np.sin(phi) * np.sin(theta)

def funcaoZ(theta, phi):
        return r * np.cos(phi)

# Calcula os pontos no R³ (x, y, z) passando as variaveis de parametrização pelas funções parametrizadas 
X = funcaoX(theta, phi)
Y = funcaoY(theta, phi)
Z = funcaoZ(theta, phi)

print(X.shape)
print(Y.shape)
print(Z.shape)

# Transforma as matrizes X, Y e Z em vetores unidimensionais 
x = X.flatten()
y = Y.flatten()
z = Z.flatten()
Z_zeros = np.zeros(z.shape) # gera um array de zeros do mesmo formato de z

print(x.shape)
print(y.shape)
print(z.shape)

# -------------------------------------------------------------------------------------------------
# GERA O GRAFICO DA SUPERFÍCIE DESTACANDO OS PONTOS DE CONTROLE

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
points = np.vstack((theta.flatten(), phi.flatten())).T

print(points)

# Calcula a triangulação com o Algoritimo de Delaunay (biblioteca Scipy)
tri = Delaunay(points)

print(tri.simplices)

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