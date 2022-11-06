#!/usr/bin/env python3
from enum import IntEnum
import numpy as np
from os.path import join
from sklearn.datasets import fetch_openml
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sys import argv

class Use(IntEnum):
    train = 0
    valid = 1
    test = 2

def main():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    print('Fetched dataset')

    split = 60000
    X /= 255.0

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clfs = [
        DecisionTreeClassifier(random_state=0),
        BaggingClassifier(random_state=0),
        RandomForestClassifier(random_state=0),
        GradientBoostingClassifier(random_state=0),
   ]

    param_grids = [
        {
         'criterion': ['gini', 'entropy'],
         'splitter': ['best', 'random'],
         'max_depth': [len(X_train), None],
        },
        {
         'n_estimators': [10, 20, 30],
         'bootstrap': [True, False]
        },
        {
         'n_estimators': [100, 200, 300],
         'criterion': ['gini', 'entropy'],
         'max_depth': [len(X_train), None],
        },
        {
         'criterion': ['friedman_mse', 'squared_error'],
         'n_estimators': [100, 200, 300],
        },
    ]

    for i in range(len(clfs)):
        clf = clfs[i]
        param_grid = param_grids[i]

        table_cols = '|c' * (len(param_grid) + 1) + '|'
        param_names = ' & '.join((param.title() for param in param_grids[i]))
        raw = f'{param_names} & Accuracy_Score \\\\'
        esc = raw.replace('_', '\\_')

        print(clf)
        print(f'\\begin{{tabular}}{{{table_cols}}}')
        print('\\hline')
        print(esc)
        print('\\hline')

        grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)

        params = ' & '.join((str(value) for value in 
                            grid_search.best_params_.values()))

        raw = f'{params} & {accuracy_score(y_test, y_pred)} \\\\'
        esc = raw.replace('_', '\\_')

        print(esc)
        print('\\hline')
        print('\\end{tabular}')

if __name__ == '__main__':
    main()
