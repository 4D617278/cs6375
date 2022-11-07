#!/usr/bin/env python3
from enum import IntEnum
import numpy as np
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

def loadtxt_dir(dir, filename):
    path = join(argv[1], filename)
    return np.loadtxt(path, np.uint8, delimiter=',')

def main():
    if len(argv) <= 1:
        print(f'usage: {argv[0]} <dir>')
        exit(1)

    num_clauses = np.array([300, 500, 1000, 1500, 1800])
    num_examples = np.array([100, 1000, 5000])

    col_index = {}
    row_index = {}

    for i, ncols in enumerate(num_clauses):
        col_index[ncols] = i
    for i, nrows in enumerate(num_examples):
        row_index[nrows] = i

    shape = (len(num_clauses), len(num_examples), len(Use))
    data = np.empty(shape=shape, dtype=object)

    for c in num_clauses:
        for e in num_examples:
            col = col_index[c]
            row = row_index[e]
            suffix = f'_c{c}_d{e}.csv'

            for use in Use:
                data[col][row][use] = loadtxt_dir(argv[1], use.name + suffix)


    clfs = [
        DecisionTreeClassifier(random_state=0),
        BaggingClassifier(random_state=0),
        RandomForestClassifier(random_state=0),
        GradientBoostingClassifier(random_state=0),
    ]

    param_grids = [
        {
         'criterion': ['gini', 'entropy'],
         'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
         'splitter': ['best', 'random'],
        },
        {
         'bootstrap_features': [False, True],
         'n_estimators': [100, 150, 200],
        },
        {
         'max_depth': [4, 6, 7, 8, 9, 10, 11, 12],
         'n_estimators': [100, 150, 200, 250, 300, 350],
        },
        {
         'loss': ['log_loss', 'exponential'],
         'learning_rate': [0.1, 0.2, 0.3],
         'n_estimators': [100, 200, 300],
        }
    ]

    metrics = [accuracy_score, f1_score]

    metric_strs = ' & '.join(metric.__name__.title() for metric in metrics)

    for i in range(len(clfs)):
        clf = clfs[i]
        param_grid = param_grids[i]

        table_cols = '|c' * (len(param_grid) + len(metrics) + 1) + '|'
        param_names = ' & '.join((param.title() for param in param_grids[i]))
        raw = f'Dataset & {param_names} & {metric_strs} \\\\'
        esc = raw.replace('_', '\\_')

        print(clf)
        print(f'\\begin{{tabular}}{{{table_cols}}}')
        print('\\hline')
        print(esc)
        print('\\hline')

        for c in range(data.shape[0]):
            for r in range(data.shape[1]):
                train = data[c][r][Use.train]
                y = data[c][r][Use.test]
                X = np.concatenate((train, data[c][r][Use.valid]))

                test_fold = np.zeros(len(X))
                test_fold[:len(train)] = -1
                cv = PredefinedSplit(test_fold)

                grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=cv)
                grid_search.fit(X[:, :-1], X[:, -1])
                y_pred = grid_search.predict(y[:, :-1])

                dataset = f'c{num_clauses[c]}_d{num_examples[r]}'
                params = ' & '.join((str(value) for value in 
                                    grid_search.best_params_.values()))
                scores = ' & '.join(str(metric(y[:, -1], y_pred))
                           for metric in metrics)

                raw = f'{dataset} & {params} & {scores} \\\\'
                esc = raw.replace('_', '\\_')

                print(esc)
                print('\\hline')

        print('\\end{tabular}')

if __name__ == '__main__':
    main()
