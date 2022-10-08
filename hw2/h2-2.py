#!/usr/bin/env python3
from enum import IntEnum
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class Classifier(IntEnum):
    svc = 0
    mlp = 1
    knn = 2

def main():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    print('Fetched dataset')

    split = 60000
    X /= 255.0

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    param_grids = [
        [{'kernel': ['rbf', 'poly'],
         'degree': [2],
         'C': [1.6, 1.8, 2.0, 2.2, 2.4]}],

        [{'hidden_layer_sizes': [(100,), (200,), (300,)],
         'activation': ['logistic', 'tanh'],
         'alpha': [1e-4, 1e-3]}],

        [{'n_neighbors': [5, 6], 
         'weights': ['distance'],
         'algorithm': ['ball_tree'],
         'leaf_size': [20, 30, 40],
         'metric': ['minkowski', 'l1']}]
    ]

    estimators = [SVC(), MLPClassifier(), KNeighborsClassifier()]

    for c in Classifier:
        grid_search = GridSearchCV(estimators[c], param_grids[c], 
                                   n_jobs=-1, verbose=3, cv=2)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        print(grid_search.score(X_test, y_test))

if __name__ == '__main__':
    main()
