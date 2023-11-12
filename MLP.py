import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main():
    print("Loading data...")
    tr = np.loadtxt('treinamento.txt')
    ts = np.loadtxt('teste.txt')
    y_test = ts[:, 132]
    y_train = tr[:, 132]
    X_train = tr[:, 1: 132]
    X_test = ts[:, 1: 132]

# MLP
    print("Scaling data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    print("Training MLP...")
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10), random_state=1) #0.94
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1) #0.93
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,100,100), random_state=1) #0.95
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(500,500,500,500), random_state=1) #0.96
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500,500,500,500), random_state=1) #0.95
    # clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(500,500,500,500), random_state=1) #0.95
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(500, 500, 500, 500), random_state=1)  # 0.95

    print("Fitting MLP (this may take a while)...")
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(classification_report(y_test, clf.predict(X_test)))


if __name__ == "__main__":
    main()
