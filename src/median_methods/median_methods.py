
import numpy as np
#import statistics
import math
import scipy.sparse as sp
from zlib import crc32
random = np.random.RandomState(crc32(str.encode(__file__)))
from abc import ABC, abstractmethod #abstract classes

#abstract
class IterativeMethod(ABC):

	def __init__(self, A, b, start):
		self.A = A
		self.b = b
		self.start = start
		self.guess = start
		self.rows, self.cols = self.A.shape
		self.row_idx = None #The index of the last row that was sampled
		#ADD: normalize rows of A

	@abstractmethod
	def sample_row_idx(self):
		pass

	@abstractmethod
	def next_iterate(self, row_idx, guess):
		pass

	def do_iteration(self):
		self.row_idx = self.sample_row_idx()
		self.guess = self.next_iterate(self.row_idx, self.guess)

	#offset to the hyperplane indexed by last sampled index
	#TODO? Could avoid recomputing this value if desired
	def cur_offset_to_hyperplane(self):
		return self.offset_to_hyperplane(self.row_idx)

	def offset_to_hyperplane(self, idx):
		return np.dot(self.A[idx], self.guess) - self.b[idx]

	def currentGuess(self):
		return self.guess

	def distanceTo(self, soln):
		return np.linalg.norm(np.reshape(soln, self.cols) - self.guess)

#abstract
class UniformRowMethod(IterativeMethod):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	def sample_row_idx(self):
		return np.random.randint(0,self.rows)

#Abstract
class ThresholdedRK(UniformRowMethod):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	@abstractmethod
	def threshold(self):
		pass

	def next_iterate(self, idx, guess):
		d = self.offset_to_hyperplane(idx)
		if(abs(d) <= self.threshold()):
			return guess - d * self.A[idx]
		else:
			return guess

#abstract
class QuantileMethod(IterativeMethod):

	def __init__(self, A, b, start, *, quantile):
		super().__init__(A,b,start)
		self.quantile = quantile

	@abstractmethod
	def get_quantile(self):
		pass

#abstract
class SGDMethod(UniformRowMethod):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	@abstractmethod
	def step_size(self):
		pass

	'''Partial subgradient for the l1 objective'''
	#def l1_gradient(self, x, a_i, b_i):
	#	ret = np.sign(np.dot(a_i,x)-b_i) * a_i
	#	return ret

	def next_iterate(self, idx, guess):
		a = self.A[idx]
		return guess - self.step_size() * np.sign(np.dot(a,guess) - self.b[idx]) * a


#TO DO: Avoid the initial "lag"
#Sliding window approach
#Abstract
class SWQuantileMethod(QuantileMethod):

	def __init__(self, A, b, start, *, quantile, window_size):
		super().__init__(A,b,start,quantile=quantile)
		self.window_size = window_size
		self.window = [0]

	#fix this to use a quantile
	def get_quantile(self):
		return np.quantile(self.window[-self.window_size:], self.quantile)

	#override
	def do_iteration(self):
		super().do_iteration()
		self.window.append(abs(self.cur_offset_to_hyperplane())) #clean this up perhaps

#Subsample the rows with replacement
#Abstract
class SampledQuantileMethod(QuantileMethod):

	def __init__(self, A, b, start, *, quantile, samples):
		super().__init__(A,b,start,quantile=quantile)
		self.samples = samples

	#update to give quantile
	def get_quantile(self):
		distances = []
		#not very pythonic...
		for i in range(0,self.samples):
			rand_idx = np.random.randint(0,self.rows)
			d = abs(self.offset_to_hyperplane(rand_idx))
			distances.append(d)
		return np.quantile(distances,self.quantile)


##### non-abstract ############
class RK(ThresholdedRK):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	def threshold(self):
		return math.inf

class SampledQuantileRK(SampledQuantileMethod, ThresholdedRK):

	def __init__(self, A, b, start, *, quantile, samples):
		SampledQuantileMethod.__init__(self, A,b,start,quantile=quantile,samples=samples)

	def threshold(self):
		return self.get_quantile()

class SWQuantileRK(SWQuantileMethod, ThresholdedRK):

	def __init__(self, A, b, start, *, quantile, window_size):
		SWQuantileMethod.__init__(self, A, b, start, quantile=quantile, window_size=window_size)

	def threshold(self):
		return self.get_quantile()

class SW_SGD(SWQuantileMethod, SGDMethod):

	def __init__(self, A, b, start, *, quantile, window_size):
		SWQuantileMethod.__init__(self, A, b, start, quantile=quantile, window_size=window_size)

	def step_size(self):
		return self.get_quantile()

class Fixed_Step_SGD(SGDMethod):

	def __init__(self, A, b, start, *, step_size):
		super().__init__(self, A, b, start)
		self.step_size = step_size

	def step_size(self):
		return self.step_size


########### end classes ##############



#SGD l1 minimization with the optimal step size chosen at each iteration.
#optimal means that the step size is chosen to minimize the expected
#squared norm of the error after the next iteration.  
#(One could argue that the optimal algorithm should really minimize the
#norm of the error rather than the squared norm.)
#this algorithm cheats by using the actual solution.
#Very slow!
def OPT_SGD(A, b, iters, soln):
	rows, cols = A.shape
	guess = np.zeros(cols)
	errors = []

	def opt_step_size(x):
		avg = (1/rows)*sum([gr(x, A[i], b[i]) for i in range(0,rows)])
		row_soln = np.reshape(soln, cols)
		e = x - row_soln
		return np.dot(e, avg)

	for i in range(0,iters):

		eta = opt_step_size(guess)
		guess = SGD_iteration(A,b,guess,eta)

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error/first_error)


	return guess, errors
