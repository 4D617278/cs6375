from math import log
from h1 import Class

def multi_prob(vector, args):
	priors, cond_probs = args
	probs = [None for c in Class]
	for c in Class:
		probs[c] = log(priors[c])
		cond_prob = cond_probs[c]
		for word in vector.keys():
			if word in cond_prob:
				probs[c] += log(cond_prob[word])
			else:
				probs[c] += log(cond_prob[None])
	return probs.index(max(probs))

def bin_prob(vector, args):
	priors, cond_probs = args
	probs = [None for c in Class]
	for c in Class:
		probs[c] = log(priors[c])
		cond_prob = cond_probs[c]
		for word in set(vector):
			if word in cond_prob:
				probs[c] += log(cond_prob[word])
			else:
				probs[c] += log(cond_prob[None])
		#for word in cond_prob.keys():
		#	if word in set(vector):
		#		probs[c] += log(cond_prob[word])
		#	else:
		#		probs[c] += log(1 - cond_prob[word])
	return probs.index(max(probs))

# add-1 smoothing
def smooth(counts, total, cond_probs):
	for word in counts:
		cond_probs[word] = (counts[word] + 1) / total

def train_multi(matrix, cond_probs):
	for c in Class:
		total = matrix[c].total() + len(matrix[c].keys())
		smooth(matrix[c], total, cond_probs[c])

def train_bin(num_files, matrix, cond_probs):
	for c in Class:
		total = num_files[c] + 2 # num subsets
		smooth(matrix[c], total, cond_probs[c])
