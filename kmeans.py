from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import time

# Carrega dados
print("Loading data...")
tr = np.loadtxt('treinamento.txt')
ts = np.loadtxt('teste.txt')

# Seleciona classes
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Filtra instâncias e rótulos
xc = []
yc = []

for c in classes:
    idxs = np.nonzero(tr[:, 132] == c)[0]
    xc.append(tr[idxs[:100], 1:132])  # Limita a 100 instâncias de cada classe
    yc.extend([c] * min(100, len(idxs)))

xc = np.concatenate(xc)
yc = np.array(yc)

# Divide dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(xc, yc, test_size=0.2)

# Normaliza os dados
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)  # PCA para redução de dimensionalidade
pca = PCA(n_components=2)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)


# PCA para redução de dimensionalidade
pca = PCA(n_components=2)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# Visualiza as instâncias de treinamento no espaço PCA
for digit_class in sorted(list(set(y_train))):
    indexes = y_train == digit_class
    plt.scatter(x_train_pca[indexes, 0],
                x_train_pca[indexes, 1], label=str(digit_class))
plt.legend()

# KMeans para geração de centróides
num_centroids_list = [1, 5, 10, 20]

for num_centroids in num_centroids_list:
    centroids = np.zeros((len(classes) * num_centroids, x_train_pca.shape[1]))
    labels = np.zeros(len(classes) * num_centroids)

    for i, digit_class in enumerate(classes):
        indexes = y_train == digit_class
        kmeans = KMeans(n_clusters=num_centroids,
                        n_init=10).fit(x_train_pca[indexes])
        centroids[i*num_centroids:(i+1)*num_centroids,
                  :] = kmeans.cluster_centers_
        labels[i*num_centroids:(i+1)*num_centroids] = digit_class

    # Treino do modelo KNN com os centróides
    # Ajustando n_neighbors para ser no máximo o número de instâncias
    knn = KNeighborsClassifier(n_neighbors=min(5, len(labels)))
    start_time = time.time()
    knn.fit(centroids, labels)
    training_time = time.time() - start_time

    # Avaliação no conjunto de teste
    y_pred = knn.predict(x_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Num Centroids: {num_centroids}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.4f} seconds\n")

    # Visualização dos centróides no espaço PCA
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100,
                marker='x', label=f'Centroids (k={num_centroids})')

plt.legend()
plt.show()
