
import numpy as np
from abc import ABC, abstractmethod 

#partial gradient used for l1 methods
def l1_partial_grad(A_i, b_i, x):
	return np.sign(np.dot(A_i,x) - b_i) * A_i

############ Abstract Classes ############

class IterativeMethod(ABC):

	def __init__(self, A, b, start):
		self.A = A
		self.b = b
		self.start = start
		self.guess = start
		self.rows, self.cols = self.A.shape
		self.row_idx = None #The index of the last row that was sampled
		#TODO: normalize rows of A

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

class UniformRowMethod(IterativeMethod):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	def sample_row_idx(self):
		return np.random.randint(0,self.rows)

class ThresholdedRKMethod(UniformRowMethod):

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

class QuantileMethod(IterativeMethod):

	def __init__(self, A, b, start, *, quantile):
		super().__init__(A,b,start)
		self.quantile = quantile

	@abstractmethod
	def get_quantile(self):
		pass

class OptStepMethod(IterativeMethod):

	def __init__(self, A, b, start, *, soln):
		super().__init__(A,b,start)
		self.soln = soln

	def opt_step_size(self):
		avg = (1/self.rows)*sum([l1_partial_grad(self.A[i], self.b[i], self.guess) for i in range(0,self.rows)])
		row_soln = np.reshape(self.soln, self.cols)
		e = self.guess - row_soln
		return np.dot(e, avg)

class SGDMethod(UniformRowMethod):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	@abstractmethod
	def step_size(self):
		pass

	def next_iterate(self, idx, guess):
		grad = l1_partial_grad(self.A[idx], self.b[idx], guess)
		return guess - self.step_size() * grad


'''Sliding window approach'''
class SWQuantileMethod(QuantileMethod):

	def __init__(self, A, b, start, *, quantile, window_size):
		super().__init__(A,b,start,quantile=quantile)
		self.window_size = window_size
		self.window = [0]

	#Note: uses a smaller window until enough samples are taken
	def get_quantile(self):
		return np.quantile(self.window[-self.window_size:], self.quantile)

	#override
	def do_iteration(self):
		super().do_iteration()
		self.window.append(abs(self.cur_offset_to_hyperplane())) #clean this up perhaps

'''Subsample the rows with replacement'''
class SampledQuantileMethod(QuantileMethod):

	def __init__(self, A, b, start, *, quantile, samples):
		super().__init__(A,b,start,quantile=quantile)
		self.samples = samples

	def get_quantile(self):
		sampled_indices = np.random.randint(self.rows, size=self.samples)
		distances = [abs(self.offset_to_hyperplane(i)) for i in sampled_indices]
		return np.quantile(distances,self.quantile)


############## Concrete Classes ##############
class RK(ThresholdedRKMethod):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	def threshold(self):
		return np.inf

class SampledQuantileRK(SampledQuantileMethod, ThresholdedRKMethod):

	def __init__(self, A, b, start, *, quantile, samples):
		SampledQuantileMethod.__init__(self, A,b,start,quantile=quantile,samples=samples)

	def threshold(self):
		return self.get_quantile()

class SWQuantileRK(SWQuantileMethod, ThresholdedRKMethod):

	def __init__(self, A, b, start, *, quantile, window_size):
		SWQuantileMethod.__init__(self, A, b, start, quantile=quantile, window_size=window_size)

	def threshold(self):
		return self.get_quantile()

class SW_SGD(SWQuantileMethod, SGDMethod):

	def __init__(self, A, b, start, *, quantile, window_size):
		SWQuantileMethod.__init__(self, A, b, start, quantile=quantile, window_size=window_size)

	def step_size(self):
		return self.get_quantile()

class FixedStepSGD(SGDMethod):

	def __init__(self, A, b, start, *, eta):
		super().__init__(A, b, start)
		self.eta = eta

	def step_size(self):
		return self.eta

class OptSGD(OptStepMethod, SGDMethod):

	def __init__(self, A, b, start, *, soln):
		OptStepMethod.__init__(self, A,b,start, soln=soln)

	def step_size(self):
		return self.opt_step_size()
