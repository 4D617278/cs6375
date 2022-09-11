#!/usr/bin/env python3
from collections import Counter
from enum import Enum
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
def cond_prob(_class, priors, word_sets, cond_probs):
	prob = 0
	for word in word_sets:
		prob += log(cond_probs[_class][word])
	prob += log(priors[_class])
	return prob

# c \in Class 
# c_max | P(c_max | d) >= P(c | d) forall c \in Class
def max_prob(vector, priors, word_sets, cond_probs) -> Class:
	return max([cond_prob(c, priors, word_sets, cond_probs) for c in Class])

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

	# Train

	# w_cni, c -> class, example number, word index
	# [ [{w_111: |w_111|, ..., w_11n: |w_11n|}, ..., {w_1n1: |w_1n1|, ..., w_1nn: |w_1nn|}], 
	#   [{w_211: |w_211|, ..., w_21n: |w_21n|}, ..., {w_2n1: |w_2n1|, ..., w_2nn: |w_2nn|}] ]
	matrix = [[] for c in Class]

	# [{w_11: |w_11|, w_1n: |w_1n|}, {w_21: |w_21|, ..., w_2n: |w_2n|}]
	sum_counts = [Counter() for c in Class]

	add_data(matrix, sum_counts, train_dirs)

	df = pd.DataFrame(matrix)
	counts = df.count(axis='columns')
	priors = counts.div(counts.sum(), axis='rows')

	# [{w_11: f_11, w_1n: f_1n}, {w_21: f_21, ..., w_2n: f_2n}]
	cond_probs = [{} for c in Class]

	# add-1 smoothing
	for i in range(len(Class)):
		total = sum_counts[i].total() + len(sum_counts[i].keys())
		for word in sum_counts[i]:
			cond_probs[i][word] = (sum_counts[i][word] + 1) / total

	print(cond_probs)

	# Test
	#add_data(matrix, sum_counts, test_dirs)
	#for vector in matrix:
	#	max_prob(vector, priors, sum_counts, cond_probs)

def add_counts(vector, sum_counts, file):
	f = open(file, 'r', errors='replace')
	text = f.read()
	f.close()

	counts = Counter(word_tokenize(text))
	sum_counts.update(counts)
	vector.append(counts)

def add_data(matrix, sum_counts, dirs):
	for i in range(len(Class)):
		for dir in dirs[i]:
			for dirpath, _, filenames in walk(dir.rstrip()):
				for filename in filenames:
					file = join(dirpath, filename)
					add_counts(matrix[i], sum_counts[i], file)

if __name__ == '__main__':
	main()
