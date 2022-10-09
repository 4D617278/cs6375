#!/usr/bin/env python3
from enum import IntEnum
from gc import collect
import numpy as np
from sys import argv

class Use(IntEnum):
	train = 0
	test = 1

START_INDEX = 1
NUM_MOVIES = 1821
NUM_USERS = 28978

def main():
    if len(argv) < START_INDEX + len(Use):
        args = ' '.join(u.name for u in Use)
        print(f'usage: {argv[0]} {args}')
        exit(1)

    data = [np.zeros(shape=(NUM_USERS, NUM_MOVIES), dtype=np.uint8)] * len(Use)

    for u in Use:
        with open(argv[START_INDEX + u], 'r') as f:
            lines = f.read().splitlines()
            lines = [line.split(',') for line in lines]
            movie_ids = set([int(line[0]) for line in lines])
            user_ids = set([int(line[1]) for line in lines])
            movie_indices = {v: k for k, v in enumerate(set(movie_ids))}
            user_indices = {v: k for k, v in enumerate(set(user_ids))}

            for line in lines:
                movie_id = int(line[0])
                user_id = int(line[1])
                vote = int(float(line[2]))
                data[u][user_indices[user_id]][movie_indices[movie_id]] = vote

    row_means = data[Use.train].mean(axis=1)
    mean_diffs = data[Use.train] - row_means[:, np.newaxis]
    weights = np.corrcoef(data[Use.train])
    print('Calculated weights')

    # normalize
    for user in range(weights.shape[0]):
        weights[user] /= np.fabs(weights[user]).sum()
    print('Normalized weights')

    out = row_means[:, np.newaxis] + weights @ mean_diffs
    print(out.shape)
    print(f'Mean Absolute Error: {np.mean(np.abs(out - data[Use.test]))}')
    rmse = np.sqrt(np.mean((out - data[Use.test]) ** 2))
    print(f'Root Mean Squared Error: {rmse}')

if __name__ == '__main__':
    main()
