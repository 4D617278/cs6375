#!/usr/bin/env python3
import nltk
from os import walk
from os.path import join
import sys

def main():
	if len(sys.argv) < 2:
		print(f'usage: {sys.argv[0]} <filename>')
		exit(1)

	for dirpath, dirnames, filenames in walk(sys.argv[1]):
		data = [(join(dirpath, f), {}) for f in filenames]

	for i in range(len(data)):
		tokens = []
		with open(data[i][0], 'r', errors='replace') as f:
			words = f.read()
			tokens = nltk.word_tokenize(words)

		for token in tokens:
			if token in data[i][1]:
				data[i][1][token] += 1
			else:
				data[i][1][token] = 1

		print(data[i])

if __name__ == '__main__':
	main()
