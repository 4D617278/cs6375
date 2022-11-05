#!/usr/bin/env python3
from enum import IntEnum
import numpy as np
from os import listdir
from os.path import join
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sys import argv

class Use(IntEnum):
    train = 0
    valid = 1
    test = 2

def main():
    if len(argv) <= 1:
        print(f'usage: {argv[0]} <dir>')
        exit(1)

    cols = set()
    rows = set()

    filenames = listdir(argv[1])
    for filename in filenames:
        _, ncols, nrows = filename.split('.')[0].split('_')
        cols.add(int(ncols[1:]))
        rows.add(nrows)

    col_index = {}
    row_index = {}

    for i, ncols in enumerate(cols):
        col_index[ncols] = i
    for i, nrows in enumerate(rows):
        row_index[nrows] = i

    shape = (len(Use), len(cols), len(rows))
    data = np.empty(shape=shape, dtype=object)

    for filename in filenames:
        use, ncols, nrows = filename.split('.')[0].split('_')
        ncols = int(ncols[1:])
        path = join(argv[1], filename)
        matrix = np.loadtxt(path, np.uint8, delimiter=',')
        data[Use[use]][col_index[ncols]][row_index[nrows]] = matrix

    clf = DecisionTreeClassifier()

    param_grid = [
        {'criterion': ['gini', 'entropy'],
         'splitter': ['best', 'random'],
         'max_depth': [max(cols), None],
        }
    ]

    train = data[Use.train]
    valid = data[Use.valid]
    test = data[Use.test]

    for c in range(data.shape[1]):
        for r in range(data.shape[2]):
            X = np.concatenate((train[c][r], valid[c][r]))

            test_fold = [-1] * len(train[c][r]) + [0] * len(valid[c][r])
            split = PredefinedSplit(test_fold)

            grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=split)
            grid_search.fit(X[:, :-1], X[:, -1])
            y_pred = grid_search.predict(test[c][r][:, :-1])

            print(accuracy_score(test[c][r][:, -1], y_pred))
            print(f1_score(test[c][r][:, -1], y_pred))
            print(grid_search.best_params_)

if __name__ == '__main__':
    main()
