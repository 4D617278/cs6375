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

def max_prob(vector, priors, cond_probs, prob, cls):
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

	# list of list of dicts
	# cond_prob = [class][file][word]
	cond_probs = [{} for c in Class]

	for c in Class:
		multi_cond_prob(unigram_matrix[c], cond_probs[c])

	for c in Class:
		s = 0
		for vector in test_matrix[c]:
			s += max_prob(vector, priors, cond_probs, multi_prob, c)
		print(s / len(test_matrix[c]))

	cond_probs = [{} for c in Class]

	for c in Class:
		bin_cond_prob(num_files[c], binary_matrix[c], cond_probs[c])

	for c in Class:
		s = 0
		for vector in test_matrix[c]:
			s += max_prob(set(vector), priors, cond_probs, bin_prob, c)
		print(s / len(test_matrix[c]))


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
