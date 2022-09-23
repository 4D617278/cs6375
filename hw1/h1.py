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
from sys import argv

class Class(IntEnum):
	ham = 0
	spam = 1

class Use(IntEnum):
	train = 0
	test = 1

def split(matrix, learn_matrix, dev_matrix, percent, num_files):
	for c in Class:
		split = round(percent * num_files[c])
		learn_matrix[c] = matrix[c][:split]
		dev_matrix[c] = matrix[c][split:]

def test(alg, matrix, args):
	counts = np.zeros(shape=(len(Class), len(Class)))
	for cls in Class:
		for vector in matrix[cls]:
			out = alg(vector, args)
			counts[out][cls] += 1

	precision = counts[Class.spam][Class.spam] / counts[Class.spam].sum()
	recall = counts[Class.spam][Class.spam] / counts.sum(axis=1)[Class.spam]
	print(f'Algorithm: {alg.__name__}')
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

	# binary and unigram matrix
	binary_matrix = [Counter() for c in Class]
	unigram_matrix = [Counter() for c in Class]

	for c in Class:
		for counter in matrix[Use.train][c]:
			binary_matrix[c].update(set(counter))
			unigram_matrix[c].update(counter)
		binary_matrix[c][None] = 0
		unigram_matrix[c][None] = 0

	# compute priors
	num_files = [len(matrix[Use.train][c]) for c in Class]
	total_files = sum(num_files)	
	priors = [num_files[c] / total_files for c in Class]

	# list of list of dicts
	# cond_prob = [class][file][word]
	cond_probs = [{} for c in Class]

	# multinomial
	train_multi(unigram_matrix, cond_probs)
	test(multi_prob, matrix[Use.test], (priors, cond_probs))

	cond_probs = [{} for c in Class]

	# binary
	train_bin(num_files, binary_matrix, cond_probs)
	test(bin_prob, matrix[Use.test], (priors, cond_probs))

	learn_matrix = [[] for c in Class]
	dev_matrix = [[] for c in Class]
	split(matrix[Use.train], learn_matrix, dev_matrix, 0.7, num_files)

	# log_reg
	weights = {}

	#max_weight = max(weights, key=weights.get)
	#print(f'max: {max_weight} {weight[max_weight]}')

	# weights[None] is bias
	for c in Class:
		weights.update(binary_matrix[c])

	for c in Class:
		grad_ascent(learn_matrix[c], weights, 0.1, 0.1, 0.26, c)
	test(log_reg_cls, dev_matrix, weights)

	#weights = {}
	#for c in Class:
	#	weights.update(binary_matrix[c])

	#for c in Class:
	#	grad_ascent(matrix[Use.train], weights, 0.1, 0.1, 0.1, c)
	#test(log_reg_cls, matrix[Use.test], weights)

if __name__ == '__main__':
	main()
