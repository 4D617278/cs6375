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

def test(alg, model, matrix, args, sgd=False, labels=None):
	counts = np.zeros(shape=(len(Class), len(Class)))
	if sgd:
		outputs = alg(matrix)
		for i in range(len(outputs)):
			counts[outputs[i]][args[i]] += 1
	else:
		if labels is None:
			for cls in Class:
				for vector in matrix[cls]:
					out = alg(vector, args)
					counts[out][cls] += 1
		else:
			for i in range(len(matrix)):
				out = alg(matrix[i], args)
				counts[out][labels[i]] += 1

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
	del counts['a']

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
	num_files = np.empty(shape=(len(Use), len(Class)), dtype=int)

	for u in Use:
		for c in Class:
			num_files[u][c] = len(matrix[u][c])

	priors = num_files[Use.train] / num_files[Use.train].sum()

	counts = [[Counter() for c in Class] for m in Model]
	#split = [Counter() for c in Class]

	for c in Class:
		split_index = round(num_files.sum(axis=0)[c] * 0.7)

		for counter in matrix[Use.train][c][:split_index]:
			counts[Model.bin][c].update(set(counter))
			counts[Model.multi][c].update(counter)

		for counter in matrix[Use.train][c][split_index:]:
			counts[Model.bin][c].update(set(counter))
			counts[Model.multi][c].update(counter)

		# add unknown/bias
		for m in Model:
			counts[m][c][None] = 0

	# multinomial
	#cond_probs = [{} for c in Class]
	#train_multi(counts[Model.multi], cond_probs)
	#test(multi_prob, Model.multi.name, matrix[Use.test], (priors, cond_probs))

	# binary
	#cond_probs = [{} for c in Class]
	#train_bin(num_files[Use.train], counts[Model.bin], cond_probs)
	#test(bin_prob, Model.bin.name, matrix[Use.test], (priors, cond_probs))

	inputs = [[[] for u in Use] for m in Model]
	labels = [[] for u in Use]
	vectors = [[None for u in Use] for m in Model]
	words = set()

	for u in Use:
		for c in Class:
			for counter in matrix[u][c]:
				words.update(counter)

	weights = [dict.fromkeys(words, 1) for m in Model]

	word_indices = {word: i for i, word in enumerate(words)}

	train = [[] for m in Model]
	dev = [[] for m in Model]
	train_labels = [[] for m in Model]
	dev_labels = [[] for m in Model]

	for u in Use:
		labels[u] = np.empty(num_files[u].sum(), dtype=int)

		start = 0
		for c in Class:
			inputs[Model.bin][u] += [Counter(set(c)) for c in matrix[u][c]]
			inputs[Model.multi][u] += matrix[u][c]
			labels[u][start:start + num_files[u][c]] = c
			start += num_files[u][c]

		for m in Model:
			vectors[m][u] = np.zeros(shape=(len(inputs[m][u]), len(words)))

			split_index = round(len(inputs[m][u]) * 0.7)
			counters = inputs[m][u]

			for i in range(split_index):
				for word in counters[i]:
					index = word_indices[word]
					vectors[m][u][i][index] = counters[i][word]

			for i in range(split_index, len(inputs[m][u])):
				for word in counters[i]:
					index = word_indices[word]
					vectors[m][u][i][index] = counters[i][word]

	for m in Model:
		split_index = round(len(inputs[m][Use.train]) * 0.7)
		train[m] = inputs[m][Use.train][:split_index]
		dev[m] = inputs[m][Use.train][split_index:]
		split_index = round(labels[Use.train].size * 0.7)
		train_labels[m] = labels[Use.train][:split_index]
		dev_labels[m] = labels[Use.train][split_index:]

	#print(dev)
	#print(train)

	# log_reg
	for m in Model:
		grad_ascent(train[m], train_labels[m], weights[m], 0.3, 0, 3.4)
		test(log_reg_cls, m.name, dev[m], weights[m], labels=dev_labels[m])
		grad_ascent(inputs[m][Use.train], labels[Use.train], weights[m], 0.3, 0, 3.4)
		test(log_reg_cls, m.name, matrix[Use.test], weights[m])

	# SGDClassifier
	#clf = make_pipeline(StandardScaler(), 
	#					SGDClassifier(max_iter=1E3, tol=1E-3))
	#param_grid = [
	#	{'sgdclassifier__loss': ['log_loss', 'hinge'], 
	#	 'sgdclassifier__penalty': ['l1', 'l2'],
	#	 'sgdclassifier__tol': [1E-2, 1E-3, 1E-4]},
	#]

	#grid_search = GridSearchCV(clf, param_grid)

	#for m in Model:
	#	grid_search.fit(vectors[m][Use.train], labels[Use.train])
	#	print(grid_search.best_params_)
	#	test(grid_search.predict, m.name, vectors[m][Use.test], labels[Use.test], sgd=True)

if __name__ == '__main__':
	main()
