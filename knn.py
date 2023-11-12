import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def main():
    print("Loading data...")
    tr = np.loadtxt('treinamento.txt')
    ts = np.loadtxt('teste.txt')
    y_test = ts[:, 132]
    y_train = tr[:, 132]
    X_train = tr[:, 1: 132]
    X_test = ts[:, 1: 132]

    # k-NN classifier
    n_neighbors = 5
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    neigh.fit(X_train, y_train)
    neigh.score(X_test, y_test)
    print(classification_report(y_test, neigh.predict(X_test)))


if __name__ == "__main__":
    main()
