#!/usr/bin/env python3
from collections import Counter 
from enum import IntEnum
from log_reg import grad_ascent, log_reg_cls
from math import log
from multi_bin import *
from nltk import word_tokenize
import numpy as np
from os import listdir
from os.path import join
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import GridSearchCV
from sys import argv

class Class(IntEnum):
	ham = 0
	spam = 1

class Use(IntEnum):
	train = 0
	test = 1

class Model(IntEnum):
	bin = 0
	multi = 1

def split(matrix, learn_matrix, dev_matrix, percent, num_files):
	for c in Class:
		split = round(percent * num_files[c])
		learn_matrix[c] = matrix[c][:split]
		dev_matrix[c] = matrix[c][split:]

def test(alg, model, matrix, args, sgd=False):
	counts = np.zeros(shape=(len(Class), len(Class)))
	if sgd:
		outputs = alg(matrix)
		for i in range(len(outputs)):
			counts[outputs[i]][args[i]] += 1
	else:
		for cls in Class:
			for vector in matrix[cls]:
				out = alg(vector, args)
				counts[out][cls] += 1

	precision = counts[Class.spam][Class.spam] / counts[Class.spam].sum()
	recall = counts[Class.spam][Class.spam] / counts.sum(axis=1)[Class.spam]
	print(f'Algorithm: {alg.__name__}')
	print(f'Model: {model}')
	print(f'Recall: {recall}')
	print(f'Precision: {precision}')
	print(f'Accuracy: {np.trace(counts) / np.concatenate(counts).sum()}')
	print(f'F1: {(2 * recall * precision) / (recall + precision)}')
	print()

def add_counts(vector, file):
	with open(file, 'r', errors='replace') as f:
		text = f.read()

	tokens = word_tokenize(text)
	counts = Counter(tokens)

	# preprocessing
	del counts[':']
	del counts['Subject']
	del counts['.']
	del counts['to']
	del counts[',']
	del counts['-']
	del counts['the']
	del counts['and']

	vector.append(counts)

def add_data(vector, dirs):
	for dir in dirs:
		for filename in listdir(dir):
			file = join(dir, filename)
			add_counts(vector, file)

START_INDEX = 1

def main():
	if len(argv) < START_INDEX + len(Class):
		args = ' '.join(c.name for c in Class)
		print(f'usage: {argv[0]} {args}')
		exit(1)

	# list of list of list of Counters
	# count = [Use][class][file][word]
	matrix = [[[] for c in Class] for u in Use]

	for c in Class:
		path = argv[START_INDEX + c]
		with open(path) as f:
			dirs = f.read().splitlines()

		if len(dirs) < len(Use):
		    print(f'bad file: {path}')
		    exit(1)

		for u in Use:
			add_data(matrix[u][c], dirs[u].split())

	# compute priors
	num_files = [len(matrix[Use.train][c]) for c in Class]
	total_files = sum(num_files)	
	priors = [num_files[c] / total_files for c in Class]

	counts = [[Counter() for c in Class] for m in Model]
	#split = [Counter() for c in Class]

	for c in Class:
		split_index = round(num_files[c] * 0.7)

		for counter in matrix[Use.train][c][:split_index]:
			counts[Model.bin][c].update(set(counter))
			counts[Model.multi][c].update(counter)

		for counter in matrix[Use.train][c][split_index:]:
			counts[Model.bin][c].update(set(counter))
			counts[Model.multi][c].update(counter)

		# add unknown/bias
		for m in Model:
			counts[m][c][None] = 0

	train = [[] for c in Class]
	dev = [[] for c in Class]
	split(matrix[Use.train], train, dev, 0.7, num_files)

	weights = [{} for m in Model]
	for c in Class:
		for m in Model:
			weights[m].update(counts[m][c])

	inputs = [[] for u in Use]
	labels = [[] for u in Use]
	vectors = [None for u in Use]
	words = set()

	for u in Use:
		for c in Class:
			for counter in matrix[u][c]:
				words.update(counter)

	word_indices = {word: i for i, word in enumerate(words)}

	for u in Use:
		for c in Class:
			inputs[u] += matrix[u][c] 
			labels[u] += [c for i in range(len(matrix[u][c]))]

		vectors[u] = np.zeros(shape=(len(inputs[u]), len(words)))

		split_index = round(len(inputs[u]) * 0.7)
		counters = inputs[u]

		for i in range(split_index):
			for word in counters[i]:
				index = word_indices[word]
				vectors[u][i][index] = counters[i][word]

		for i in range(split_index, len(inputs[u])):
			for word in counters[i]:
				index = word_indices[word]
				vectors[u][i][index] = counters[i][word]

	#for m in Model:
	#	max_weight = max(weights[m], key=weights[m].get)
	#	print(f'max: {max_weight} {weight[m][max_weight]}')

	# multinomial
	#cond_probs = [{} for c in Class]
	#train_multi(counts[Model.multi], cond_probs)
	#test(multi_prob, Model.multi.name, matrix[Use.test], (priors, cond_probs))

	# binary
	#cond_probs = [{} for c in Class]
	#train_bin(num_files, counts[Model.bin], cond_probs)
	#test(bin_prob, Model.bin.name, matrix[Use.test], (priors, cond_probs))

	# log_reg
	for m in Model:
		for c in Class:
			grad_ascent(train[c], weights[m], 0.1, 0, 0.26, c)
		test(log_reg_cls, m.name, dev, weights[m])
		for c in Class:
			grad_ascent(matrix[Use.train][c], weights[m], 0.1, 0, 0.26, c)
		test(log_reg_cls, m.name, matrix[Use.test], weights[m])

	# SGDClassifier
	clf = make_pipeline(# StandardScaler(), 
						SGDClassifier(max_iter=1E3, tol=1E-3))
	param_grid = [
		{'sgdclassifier__loss': ['log_loss', 'hinge'], 
		 'sgdclassifier__penalty': ['l1', 'l2']},
	]

	grid_search = GridSearchCV(clf, param_grid)
	grid_search.fit(vectors[Use.train], labels[Use.train])
	print(grid_search.best_params_)

	#for m in Model:
	#	test(clf.predict, m.name, )
	test(grid_search.predict, "", vectors[Use.test], labels[Use.test], True)

if __name__ == '__main__':
	main()
