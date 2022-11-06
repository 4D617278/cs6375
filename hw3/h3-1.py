#!/usr/bin/env python3
from enum import IntEnum
import numpy as np
from os import listdir
from os.path import join
from sklearn.ensemble import *
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

    index_col = {}
    index_row = {}

    for i, ncols in enumerate(cols):
        col_index[ncols] = i
        index_col[i] = ncols
    for i, nrows in enumerate(rows):
        row_index[nrows] = i
        index_row[i] = nrows

    shape = (len(Use), len(cols), len(rows))
    data = np.empty(shape=shape, dtype=object)

    for filename in filenames:
        use, ncols, nrows = filename.split('.')[0].split('_')
        ncols = int(ncols[1:])
        path = join(argv[1], filename)
        matrix = np.loadtxt(path, np.uint8, delimiter=',')
        data[Use[use]][col_index[ncols]][row_index[nrows]] = matrix

    clfs = [
            DecisionTreeClassifier(), 
            BaggingClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
           ]

    param_grids = [
        {
         'criterion': ['gini', 'entropy'],
         'splitter': ['best', 'random'],
         'max_depth': [max(cols), None],
        },
        {
         'n_estimators': [10, 20, 30],
         'bootstrap': [True, False]
        },
        {
         'n_estimators': [100, 200, 300],
         'criterion': ['gini', 'entropy'],
         'max_depth': [max(cols), None],
        },
        {
         'loss': ['log_loss', 'exponential'],
         'learning_rate': [0.1, 0.2, 0.3],
         'n_estimators': [100, 200, 300],
        }
    ]

    train = data[Use.train]
    valid = data[Use.valid]
    test = data[Use.test]

    for i in range(len(clfs)):
        print(clfs[i])
        table_cols = '|c' * (len(param_grids[i]) + 3) + '|'
        print(f'\\begin{{tabular}}{{{table_cols}}}')
        print('\\hline')
        params = ' & '.join((param.title() for param in param_grids[i]))
        params = params.replace('_', '\\_')
        print(f'Dataset & {params} & F1 & Accuracy \\\\')
        print('\\hline')
        for c in range(data.shape[1]):
            for r in range(data.shape[2]):
                X = np.concatenate((train[c][r], valid[c][r]))

                test_fold = [-1] * len(train[c][r]) + [0] * len(valid[c][r])
                split = PredefinedSplit(test_fold)

                grid_search = GridSearchCV(clfs[i], param_grids[i], n_jobs=-1, cv=split)
                grid_search.fit(X[:, :-1], X[:, -1])
                y_pred = grid_search.predict(test[c][r][:, :-1])

                dataset = f'c{index_col[c]}_{index_row[r]}'
                params = ' & '.join((str(value) for value in grid_search.best_params_.values()))
                acc = accuracy_score(test[c][r][:, -1], y_pred).round(2)
                f1 = f1_score(test[c][r][:, -1], y_pred)
                raw = f'{dataset} & {params} & {acc} & {f1} \\\\'
                esc = raw.replace('_', '\\_')
                print(esc)
                print('\\hline')
        print('\\end{tabular}')

if __name__ == '__main__':
    main()
