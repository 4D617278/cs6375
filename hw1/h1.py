#!/usr/bin/env python3
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
def cond_prob(_class, priors, words, cond_probs):
	prob = 0
	for word in words:
		prob += log(cond_probs[_class][word])
	prob += log(priors[_class])
	return prob

# c \in Class 
# c_max | P(c_max | d) >= P(c | d) forall c \in Class
def max_prob(priors, words, cond_probs) -> Class:
	return max([cond_prob(c, priors, words, cond_probs) for c in Class])

START_INDEX = 1

def main():
	if len(sys.argv) < START_INDEX + len(Class):
		print(f'usage: {sys.argv[0]} {" ".join(c.name for c in Class)}')
		exit(1)

	dirs = []

	for i in range(len(Class)):
		with open(sys.argv[START_INDEX + i]) as f:
			dirs.append(f.readlines())

	matrix = [[] for c in Class]
	words = [set() for c in Class]

	get_data(matrix, words, dirs)
	df = pd.DataFrame(matrix)

	counts = df.count(axis='columns')
	priors = counts.div(counts.sum(), axis='rows')

	print(words)

	#max_prob()

def add_counts(vector, words, file):
	f = open(file, 'r', errors='replace')
	tokens = word_tokenize(f.read())
	f.close()

	counts = {}

	for token in tokens:
		if token in counts:
			counts[token] += 1
		else:
			counts[token] = 1

	words.update(set(tokens))
	vector.append(counts)

def get_data(matrix, words, dirs):
	for i in range(len(Class)):
		for dir in dirs[i]:
			for dirpath, _, filenames in walk(dir.rstrip()):
				for filename in filenames:
					file = join(dirpath, filename)
					add_counts(matrix[i], words[i], file)

if __name__ == '__main__':
	main()
