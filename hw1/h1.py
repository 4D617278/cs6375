#!/usr/bin/env python3
from collections import Counter
from enum import Enum
import math
from nltk import word_tokenize
from os import walk
from os.path import join
import pandas as pd
import sys

class Class(Enum):
	ham = 0
	spam = 1

# real \in [0, 1]
#) P(c | d) = log(P(c)) + sum(log P(w|c)) \forall w in D 
def prob(vector, priors, cond_probs):
	prob = math.log(priors)
	for word in vector.keys():
		if word in cond_probs:
			prob += math.log(cond_probs[word])
		else:
			prob += math.log(cond_probs[None])
	return prob

# c \in Class 
# c_max | P(c_max | d) >= P(c | d) forall c \in Class
def max_prob(vector, priors, cond_probs) -> Class:
	probs = [prob(vector, priors[c], cond_probs[c]) for c in range(len(Class))]
	max_p = max(probs)
	c_max = probs.index(max_p)
	return c_max

START_INDEX = 1

def main():
	if len(sys.argv) < START_INDEX + len(Class):
		args = ' '.join(c.name for c in Class)
		print(f'usage: {sys.argv[0]} {args}')
		exit(1)

	train_dirs = []
	test_dirs = []

	for i in range(len(Class)):
		with open(sys.argv[START_INDEX + i]) as f:
			dirs = [line.split() for line in f.readlines()]
			train_dirs.append([dir[0] for dir in dirs])
			test_dirs.append([dir[1] for dir in dirs])

	# w_cni, c -> class, vector number, word index
	# [ [{w_111: |w_111|, ..., w_11n: |w_11n|}, ..., {w_1n1: |w_1n1|, ..., w_1nn: |w_1nn|}], 
	#   [{w_211: |w_211|, ..., w_21n: |w_21n|}, ..., {w_2n1: |w_2n1|, ..., w_2nn: |w_2nn|}] ]
	train_matrix = [[] for c in Class]
	test_matrix = [[] for c in Class]

	add_data(train_matrix, train_dirs)
	add_data(test_matrix, test_dirs)

	# compute priors
	df = pd.DataFrame(train_matrix)
	file_counts = df.count(axis='columns')
	priors = file_counts.div(file_counts.sum(), axis='rows')

	# [{w_11: |w_11|, w_1n: |w_1n|}, {w_21: |w_21|, ..., w_2n: |w_2n|}]
	counts = [Counter() for c in Class]
	# [{w_11: f_11, w_1n: f_1n}, {w_21: f_21, ..., w_2n: f_2n}]
	cond_probs = [{} for c in Class]

	# train
	for i in range(len(Class)):
		for vector in train_matrix[i]:
			counts[i].update(vector)

		# add unknown
		counts[i][None] = 0

		# add-1 smoothing
		total = counts[i].total() + len(counts[i].keys())
		for word in counts[i]:
			cond_probs[i][word] = (counts[i][word] + 1) / total

	# test
	for i in range(len(Class)):
		good = 0
		for vector in test_matrix[i]:
			good += (max_prob(vector, priors, cond_probs) == i)
		print(good / len(test_matrix[i]))

def add_counts(vector, file):
	f = open(file, 'r', errors='replace')
	text = f.read()
	f.close()

	counts = Counter(word_tokenize(text))
	vector.append(counts)

def add_data(matrix, dirs):
	for i in range(len(Class)):
		for dir in dirs[i]:
			for dirpath, _, filenames in walk(dir.rstrip()):
				for filename in filenames:
					file = join(dirpath, filename)
					add_counts(matrix[i], file)

if __name__ == '__main__':
	main()
