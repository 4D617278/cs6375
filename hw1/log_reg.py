import math

def dot_prod(v1, v2):
	return sum(v1[key] * v2.get(key, 0) for key in v1)

def log_reg_cls(counter, weights):
	return int(dot_prod(counter, weights) > 0)

def softmax(dot):
	# prevent overflow
	if dot >= 0:
		exp = math.e ** -dot
		return 1 / (1 + exp)
	else:
		exp = math.e ** dot
		return exp / (1 + exp)

def grad_ascent(vector, weights, rate, penalty, max_error, cls):
	penalty_factor = (1 - rate * penalty)
	total_error = max_error
	error_sums = {}

	# add bias
	for counter in vector:
		counter[None] = 1

	#for i in range(loops):
	while abs(total_error) >= max_error:
		total_error = 0

		for weight in weights:
			error_sums[weight] = 0

		for counter in vector:
			# t_d - o_d
			dot = dot_prod(counter, weights)
			out = softmax(dot)
			#print(f'dot: {dot}, out: {out}')

			error = cls - out
			total_error += error

			# (t_d - o_d) * v_i
			for weight in weights.keys() & counter.keys():
				error_sums[weight] += error * counter[weight]

		for weight in weights:
			weights[weight] *= penalty_factor 
			weights[weight] += rate * error_sums[weight]

		#print(total_error)
