#!/usr/bin/env python3
from enum import IntEnum
import numpy as np
from os import listdir
from os.path import join
from sklearn.tree import DecisionTreeClassifier
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
        cols.add(ncols)
        rows.add(nrows)

    col_index = {}
    row_index = {}

    for i, ncols in enumerate(cols):
        col_index[ncols] = i
    for i, nrows in enumerate(rows):
        row_index[nrows] = i

    data = np.empty(shape=(len(Use), len(cols), len(rows)), dtype=object)

    for filename in filenames:
        use, ncols, nrows = filename.split('.')[0].split('_')
        path = join(argv[1], filename)
        matrix = np.loadtxt(path, np.uint8, delimiter=',')
        data[Use[use]][col_index[ncols]][row_index[nrows]] = matrix

    # print(data[Use.train][col_index['c500']][row_index['d5000']].shape)

if __name__ == '__main__':
    main()
