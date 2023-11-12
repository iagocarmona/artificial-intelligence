import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def main():
    print("Loading data...")
    tr = np.loadtxt('treinamento.txt')
    ts = np.loadtxt('teste.txt')
    y_test = ts[:, 132]
    y_train = tr[:, 132]
    X_train = tr[:, 1: 132]
    X_test = ts[:, 1: 132]


# Random Forest Classifier
    print("Training Random Forest Classifier...")
#     _, _ = make_classification(n_samples=1000, n_features=4,
#                                n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    clf = RandomForestClassifier(
        n_estimators=10000, max_depth=30, random_state=1)

    print("Fitting data...")
    clf.fit(X_train, y_train)
    print(clf.feature_importances_)
    print(clf.predict(X_test))
    print(classification_report(y_test, clf.predict(X_test)))


if __name__ == "__main__":
    main()
