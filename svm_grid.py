import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def main():
    print("Loading data...")
    tr = np.loadtxt('treinamento.txt')
    ts = np.loadtxt('teste.txt')
    y_test = ts[:, 132]
    y_train = tr[:, 132]
    X_train = tr[:, 1: 132]
    X_test = ts[:, 1: 132]

# SVM com Grid search
    C_range = 2. ** np.arange(-5, 15, 2)
    gamma_range = 2. ** np.arange(3, -15, -2)

    # instancia o classificador, gerando probabilidades
    srv = svm.SVC(probability=True, kernel='rbf')
    ss = StandardScaler()
    pipeline = Pipeline([('scaler', ss), ('svm', srv)])

    param_grid = {
        'svm__C': C_range,
        'svm__gamma': gamma_range
    }

    # faz a busca
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)

    # recupera o melhor modelo
    model = grid.best_estimator_
    print(classification_report(y_test, model.predict(X_test)))


if __name__ == "__main__":
    main()
