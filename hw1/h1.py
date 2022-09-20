#!/usr/bin/env python3
from collections import Counter
from enum import IntEnum
import math
from nltk import word_tokenize
from os import walk
from os.path import join
import sys

class Class(IntEnum):
	ham = 0
	spam = 1

class Algs(IntEnum):
	multi = 0
	binary = 1

class Use(IntEnum):
	train = 0
	test = 1

# P(c | d) = log(P(c)) + sum(log P(w|c)) forall w in D 
def multi_prob(vector, prior, cond_probs):
	prob = math.log(prior)
	for word in vector.keys():
		if word in cond_probs:
			prob += math.log(cond_probs[word])
		else:
			prob += math.log(cond_probs[None])
	return prob

def bin_prob(set, prior, cond_probs):
	prob = math.log(prior)
	for word in set:
		if word in cond_probs:
			prob += math.log(cond_probs[word])
		else:
			prob += math.log(cond_probs[None])
	#for word in cond_probs.keys():
	#	if word in set:
	#		prob += math.log(cond_probs[word])
	#	else:
	#		prob += math.log(1 - cond_probs[word])
	return prob

def dot_prod(v1, v2):
	return sum(v1 * v2 for (v1, v2) in zip(v1, v2))

def log_reg_cls(vector, weights):
	return (dot_prod(vector, weights) > 0)

def log_reg(vector, weights):
	# exp = 2 ** dot_prod(vector, weights)
	exp = math.e ** dot_prod(vector, weights)
	return exp / (1 + exp)

def log_reg(dot):
	# exp = 2 ** dot
	exp = math.e ** dot
	return exp / (1 + exp)

def grad_ascent(vector, weights, rate, penalty, max_error, cls):
	log_reg_table = {}

	num_weights = len(weights)
	penalty_factor = (1 - rate * penalty)
	total_error = max_error

	while abs(total_error) >= max_error:
		total_error = 0

		for i in range(num_weights):
			error_sum = 0

			for counter in vector:
				vec = list(counter.values())

				if i >= len(vec):
					continue

				dot = dot_prod(vec, weights)

				if dot not in log_reg_table:
					log_reg_table[dot] = log_reg(dot)
				error = cls - log_reg_table[dot]

				error_sum += error * vec[i]

			total_error += error_sum
			weights[i] *= penalty_factor 
			weights[i] += rate * error_sum

		print(total_error)

def bin_search():
	pass

def split(matrix, learn_matrix, dev_matrix, percent, num_files):
	for c in Class:
		split = round(percent * num_files[c])
		learn_matrix[c] = matrix[c][:split]
		dev_matrix[c] = matrix[c][split:]

def test(vector, priors, cond_probs, prob, cls):
	probs = [prob(vector, priors[c], cond_probs[c]) for c in Class]
	max_p = max(probs)
	c_max = probs.index(max_p)
	return c_max == cls

# add-1 smoothing
def smooth(counts, total, cond_probs):
	for word in counts:
		cond_probs[word] = (counts[word] + 1) / total

def multi_cond_prob(counts, cond_probs):
	total = counts.total() + len(counts.keys())
	smooth(counts, total, cond_probs)

def bin_cond_prob(vlen, counts, cond_probs):
	total = vlen + 2 # num subsets
	smooth(counts, total, cond_probs)

START_INDEX = 1

def main():
	if len(sys.argv) < START_INDEX + len(Class):
		args = ' '.join(c.name for c in Class)
		print(f'usage: {sys.argv[0]} {args}')
		exit(1)

	# list of list of directory paths
	# dirpath = [class][index]
	train_dirs = []
	test_dirs = []

	for c in Class:
		with open(sys.argv[START_INDEX + c]) as f:
			dirs = [line.split() for line in f.readlines()]
			train_dirs.append([dir[Use.train] for dir in dirs])
		test_dirs.append([dir[Use.test] for dir in dirs])

	# list of list of Counters
	# count = [class][file][word]
	train_matrix = [[] for c in Class]
	test_matrix = [[] for c in Class]

	for c in Class:
		add_data(train_matrix[c], train_dirs[c])
		add_data(test_matrix[c], test_dirs[c])

	# binary and unigram matrix
	binary_matrix = [Counter() for c in Class]
	unigram_matrix = [Counter() for c in Class]

	for c in Class:
		for counter in train_matrix[c]:
			binary_matrix[c].update(set(counter))
			unigram_matrix[c].update(counter)
		binary_matrix[c][None] = 0
		unigram_matrix[c][None] = 0

	# compute priors
	num_files = [len(train_matrix[c]) for c in Class]
	total_files = sum(num_files)	
	priors = [num_files[c] / total_files for c in Class]

	num_test_files = [len(test_matrix[c]) for c in Class]

	# list of list of dicts
	# cond_prob = [class][file][word]
	cond_probs = [{} for c in Class]
	for c in Class:
		multi_cond_prob(unigram_matrix[c], cond_probs[c])
	for c in Class:
		s = 0
		for vector in test_matrix[c]:
			s += test(vector, priors, cond_probs, multi_prob, c)
		print(s / num_test_files[c])

	cond_probs = [{} for c in Class]
	for c in Class:
		bin_cond_prob(num_files[c], binary_matrix[c], cond_probs[c])
	for c in Class:
		s = 0
		for vector in test_matrix[c]:
			s += test(set(vector), priors, cond_probs, bin_prob, c)
		print(s / num_test_files[c])

	learn_matrix = [[] for c in Class]
	dev_matrix = [[] for c in Class]
	split(train_matrix, learn_matrix, dev_matrix, 0.7, num_files)

	weights = [[] for c in Class]

	for c in Class:
		max_train_len = max([len(vector) for vector in train_matrix[c]])
		max_test_len = max([len(vector) for vector in test_matrix[c]])
		max_len = max(max_train_len, max_test_len)
		weights[c] = [0 for _ in range(max_len)]

	for c in Class:
		print()
		grad_ascent(learn_matrix[c], weights[c], 0.01, 0.1, 1, c)

	#penalty = bin_search(dev_matrix)
	#log_reg_train(train_matrix, weights, 0.01, penalty, 0.5)
	#test()
	

def add_counts(vector, file):
	f = open(file, 'r', errors='replace')
	text = f.read()
	f.close()

	counts = Counter(word_tokenize(text))
	vector.append(counts)

def add_data(vector, dirs):
	for dir in dirs:
		for dirpath, _, filenames in walk(dir.rstrip()):
			for filename in filenames:
				file = join(dirpath, filename)
				add_counts(vector, file)

if __name__ == '__main__':
	main()
