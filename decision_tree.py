import numpy as np
from sklearn.metrics import classification_report
from sklearn import tree


def main():
    print("Loading data...")
    tr = np.loadtxt('treinamento.txt')
    ts = np.loadtxt('teste.txt')
    y_test = ts[:, 132]
    y_train = tr[:, 132]
    X_train = tr[:, 1: 132]
    X_test = ts[:, 1: 132]

# Decision Tree
    print("Training Decision Tree...")
    clf = tree.DecisionTreeClassifier()

    print("Fitting data...")
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(classification_report(y_test, clf.predict(X_test)))
    tree.plot_tree(clf)


if __name__ == "__main__":
    main()
