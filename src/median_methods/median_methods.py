
import numpy as np
import statistics
import matplotlib.pyplot as plt
import math
import scipy.sparse as sp
from zlib import crc32
random = np.random.RandomState(crc32(str.encode(__file__)))

#abstract classes
from abc import ABC, abstractmethod

#The code that I wrote here is pretty hacky. (Please don't judge me)

########## SYSTEM GENERATION ###########
# Each method returns a triple (A,b,x).  Our goal is to solve Av = b for the pseudo-solution x.

#Credit to Jacob
def normalize(orig_vec, *, norm_mat=None, vec_norm=1):
    """
    Normalize the vector `orig_vec` in the `norm_mat` inner product norm to have
    exactly `vec_norm` as its norm. If `norm_mat` isn't given, use the
    standard Euclidean inner product.
    """

    if norm_mat is None:
        orig_norm = np.linalg.norm(orig_vec)
    else:
        orig_norm = np.sqrt((orig_vec.T @ norm_mat @ orig_vec)[0,0])

    return vec_norm * orig_vec / orig_norm

def consistent(A, *, sln_norm_mat=None, sln_norm=1):
    """
    Given a matrix `A`, this function returns a vector `b` such that the system
        A x = b
    has an exact solution `x` with norm of exactly `sln_norm`.
    """
    n_rows, n_cols = A.shape

    # Generate an exact solution in the row space of A
    prelim_x = A.T @ random.normal(0, 1.0, (n_rows, 1))

    # Scale solution to have desired norm
    x = normalize(prelim_x, norm_mat=sln_norm_mat, vec_norm=sln_norm)

    # Compute the final RHS vector
    b = A @ x

    return b, x
def adversarially_corrupted(A, *, sln_norm_mat=None, sln_norm=1,
                            corrupted_fraction=0.1):
    """
    Given a matrix `A`, this function returns a vector `b` such that the system
        A x = b
    is inconsistent, but has two consistent parts. The top `corrupted_fraction`
    fraction of equations will be consistent, as will the remaining bottom
    equations.
    """
    n_rows = A.shape[0]
    n_corrupted_rows = int(n_rows * corrupted_fraction)

    # Adversarial corruptions are internally consistent
    b1, x1 = consistent(A[0:n_corrupted_rows, :],
                        sln_norm_mat=sln_norm_mat, sln_norm=sln_norm)

    # The remaining entries are also internally consistent
    b2, x2 = consistent(A[n_corrupted_rows:, :],
                        sln_norm_mat=sln_norm_mat, sln_norm=sln_norm)

    # TODO: Allow user to specify how far apart x1 and x2 should be.

    # Make sure b1 is stacked on top of b2, since the rows of A corresponding
    # to b1 are on top.
    b = np.vstack([b1, b2])

    return b, x2

def normalized_gaussian(rows, cols):

	def normal_gauss_vect(n):
		g = np.random.normal(0,1,n)
		return (1/np.linalg.norm(g)) * g

	A = np.zeros((rows,cols))
	for i in range(0,rows):
		A[i] = normal_gauss_vect(cols)
	#x = np.random.uniform(-5, 5, (cols, 1))
	x = np.random.normal(0, 1, (cols, 1))
	b = np.matmul(A, x)
	return (A,b,x)

def normalized_gaussian_matrix(rows, cols):
	def normal_gauss_vect(n):
		g = np.random.normal(0,1,n)
		return (1/np.linalg.norm(g)) * g

	A = np.zeros((rows,cols))
	for i in range(0,rows):
		A[i] = normal_gauss_vect(cols)
	return A


#Rows of A are normalized Gaussian vectors (i.e. uniform over the unit sphere)
def normalized_gaussian_with_errors(rows, cols, errors, max_error):
	A,b,x = normalized_gaussian(rows,cols)

	bad_rows = np.random.choice(rows, errors, replace = False)
	for i in bad_rows:
		b[i] = np.random.uniform(-max_error, max_error)
	return (A,b,x)

#entries of A are initially sampled uniform in (low,high). Then each row of A is normalized.
#This can be used to generate poorly-conditioned systems.
def uniform_entries_with_errors(rows, cols, low, high, errors, max_error):
	A = np.random.uniform(low, high, (rows, cols))
	for i in range(0,rows):
		A[i] = A[i]/(np.linalg.norm(A[i]))

	x = np.random.uniform(-5, 5, (cols, 1))
	b = np.matmul(A, x)

	bad_rows = np.random.choice(rows, errors, replace = False)
	for i in bad_rows:
		b[i] = np.random.uniform(-max_error, max_error)
		#b[i] = 1
	return (A,b,x)

def bernoulli_with_errors(rows, cols, errors, max_error):
	A = np.random.binomial(1, 0.5, (rows,cols))*2 - np.ones((rows,cols));
	for i in range(0,rows):
		A[i] = A[i]/(np.linalg.norm(A[i]))
	x = np.random.uniform(-5, 5, (cols, 1))
	b = np.matmul(A, x)

	bad_rows = np.random.choice(rows, errors, replace = False)
	for i in bad_rows:
		b[i] = np.random.uniform(-max_error, max_error)
	return (A,b,x)


######### RECOVERY ALGORITHMS #########

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
	def update_iterate(self, row_idx):
		pass

	def do_iteration(self):
		self.row_idx = self.sample_row_idx()
		self.update_iterate(self.row_idx)

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

class ThresholdedRK(UniformRowMethod):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	@abstractmethod
	def threshold(self):
		pass

	def update_iterate(self, idx):
		d = self.offset_to_hyperplane(idx)
		if(abs(d) <= self.threshold()):
			self.guess = self.guess - d * self.A[idx]

class RK(ThresholdedRK):

	def __init__(self, A, b, start):
		super().__init__(A,b,start)

	def threshold(self):
		return float("inf")

#TO DO: Avoid the initial "lag"
class SWQuantileMethod(IterativeMethod):

	def __init__(self, A, b, start, window_size, quantile):
		super().__init__(A,b,start)
		self.window_size = window_size
		self.quantile = quantile
		self.window = self.window_size*[math.inf]

	#fix this to use a quantile
	def cur_quantile(self):
		return statistics.median(self.window[-self.window_size:])

	def do_iteration(self):
		super().do_iteration()
		self.window.append(abs(self.cur_offset_to_hyperplane()))

class SWQuantileRK(SWQuantileMethod, ThresholdedRK):

	def __init__(self, A, b, start, window_size, quantile):
		SWQuantileMethod.__init__(self, A, b, start, window_size, quantile)

	def threshold(self):
		return self.cur_quantile()


def errors_by_iteration(method, iters, soln):
	errors = []
	for i in range(0,iters):
		errors.append(method.distanceTo(soln))
		method.do_iteration()
	return errors 

def classTest():
	rows, cols = 50000,100
	iters = 20000
	A,b,soln = normalized_gaussian_with_errors(rows,cols,1000, max_error=1)
	initvect = np.zeros(cols)

	#rk = SWQuantileRK(A,b,initvect, 100, 0.5)
	rk = RK(A,b,initvect)
	errs = errors_by_iteration(rk, iters, soln)

	plt.plot(errs, label = 'SGD median')
	plt.show()

classTest()




#Each method returns the approximate pseudosolution, along with a list containing
#the norm of the error after each iteration.
#ROWS OF THE MATRIX ASSUMED TO BE NORMALIZED

def vanilla_RK(A, b, iters, x):
	rows, cols = A.shape
	guess = np.zeros(cols)
	errors = []
	for i in range(0,iters):
		idx = np.random.randint(0,rows)
		v = A[idx]
		c = (np.dot(v,guess) - b[idx])/(np.dot(v,v))
		guess = guess - c*v

		error = np.linalg.norm(np.reshape(x, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)
	return guess,errors


#RK with a sliding window median
#Only perform an RK projection if the distance travelled would be less
#than a running median
def RK_SW_median(A, b, iters, soln, window_size=10, alpha = 1):

	rows, cols = A.shape

	guess = np.zeros(cols)

	moves = window_size*[math.inf]
	errors = []

	for i in range(0,iters):

		idx = np.random.randint(0,rows)
		v = A[idx]
		norm_v = math.sqrt(np.dot(v,v))

		c = (np.dot(v,guess) - b[idx])/(norm_v**2)

		move_amt = abs(c)*norm_v
		sliding_median = statistics.median(moves[-window_size:])

		if(i > window_size and move_amt <= alpha*sliding_median ):
			guess = guess - c*v

		moves.append(move_amt)

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)

	return guess, errors

#RK with a sliding window quantile
#Only perform an RK projection if the distance travelled would be less
#than a certain quantile of the running set of residuals
def RK_SW_quantile(A, b, iters, soln, window_size=10, quantile = 0.1):

	rows, cols = A.shape

	guess = np.zeros(cols)

	moves = window_size*[math.inf]
	errors = []

	for i in range(0,iters):

		idx = np.random.randint(0,rows)
		v = A[idx]
		norm_v = math.sqrt(np.dot(v,v))

		c = (np.dot(v,guess) - b[idx])/(norm_v**2)

		move_amt = abs(c)*norm_v
		sliding_quantile = np.quantile(moves[-window_size:], quantile)

		if(i > window_size and move_amt <= sliding_quantile ):
			guess = guess - c*v

		moves.append(move_amt)

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)

	return guess, errors

#Median approach without using a sliding window
def pure_median(A, b, iters, soln, samples=100, alpha=1):
	rows, cols = A.shape
	guess = np.zeros(cols)
	errors = []

	def median(pos):
		distances = []
		for i in range(0,samples):
			idx = np.random.randint(0,rows)
			m = abs(np.dot(A[idx],pos) - b[idx])
			#print("m:", m)

			distances.append(m)
		return statistics.median(distances)

	for i in range(0,iters):
		idx = np.random.randint(0,rows)
		v = A[idx]
		c = np.dot(v,guess) - b[idx]
		#print("guess: ", guess)
		#print("c: ", c)
		#print("v: ", v)

		if(abs(c) <= alpha*median(guess) ):
			guess = guess - c*v

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)

	return guess, errors

def motzkin_hybrid(A,b,iters,soln,num_errs):
	rows, cols = A.shape
	guess = np.zeros(cols)
	errors = []

	def nth_largest(pos,n):
		distances = []
		for i in range(0,rows):

			#print("A[i]:", A[i])
			#print("pos:", pos)


			m = abs(np.dot(A[i],pos) - b[i])[0]
			#print("m:", m)
			distances.append(m)
		#print("distances: ", distances)
		#print("sorted: ", np.argsort(distances))
		return np.argsort(distances)[-n]

	for i in range(0,iters):
		idx = nth_largest(guess, num_errs+1)
		#print("idx:", idx)
		v = A[idx]
		c = np.dot(v,guess) - b[idx]
		#print("guess: ", guess)
		#print("c: ", c)
		#print("v: ", v)

		guess = guess - c*v

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)

	return guess, errors

#component of the gradient at x of ||Ax - b||_1 
#corresponding to a row a_i of A and an entry b_i of b.
def gr(x, a_i, b_i):
	d = np.dot(a_i,x) - b_i
	if d > 0: return a_i
	if d < 0: return -a_i
	if d == 0: return np.zeros(a.shape)

#A single iteration of SGD, where v is the current location, and eta is the step size
def SGD_iteration(A,b,v,eta):
	rows, cols = A.shape
	idx = np.random.randint(0,rows)
	a = A[idx]
	return v - eta * gr(v, a, b[idx])

#Performs stochastic gradient descent on the objective ||Ax-b||_1 
#with a fixed step size eta 
def SGD_l1_min(A, b, iters, soln, eta):
	rows, cols = A.shape
	guess = np.zeros(cols)
	errors = []

	for i in range(0,iters):

		guess = SGD_iteration(A,b,guess,eta)

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)

	return guess, errors

#Stochastic gradient descent with the step-size chosen as in the median algorithm
def SGD_median(A, b, iters, soln, samples):
	rows, cols = A.shape
	guess = np.zeros(cols)
	errors = []

	for i in range(0,iters):

		#compute median (This is ugly code.  Should probably be vectorized!)
		distances = []
		for j in range(0,samples):
			idx = np.random.randint(0,rows)
			v = A[idx]
			m = abs(np.dot(v,guess) - b[idx])
			distances.append(m)
		eta = statistics.median(distances)

		guess = SGD_iteration(A,b,guess,eta)

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)

	return guess, errors

#Stochastic gradient descent with the step-size chosen as in the median algorithm
def SGD_quantiles(A, b, iters, soln, samples, quantile = 0.1):
	rows, cols = A.shape
	guess = np.zeros(cols)
	errors = []

	for i in range(0,iters):

		#compute median (This is ugly code.  Should probably be vectorized!)
		distances = []
		for j in range(0,samples):
			idx = np.random.randint(0,rows)
			v = A[idx]
			m = abs(np.dot(v,guess) - b[idx])
			distances.append(m)
		eta = np.quantile(distances, quantile)

		guess = SGD_iteration(A,b,guess,eta)

		error = np.linalg.norm(np.reshape(soln, cols) - guess)
		if i == 0:
			first_error = error
			errors.append(1)
		else:
			errors.append(error / first_error)

	return guess, errors



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

#stochastic gradient descent on ||Ax-b||_1 with an adaptively chosen step size
def adaptive_l1_min(A, b, iters, soln, init_eta=1, gamma=0.99, target_avg=0.4, step_scaling=0.99):

	rows, cols = A.shape
	guess = np.zeros(cols)

	errors = []
	over_avg = 0
	eta = init_eta

	#The usual SGD step, but additionally returns 1 
	#depending on whether the hyperplane in question was crossed
	def step(x, a, b_i):
		over = 0
		d = np.dot(a,x) - b_i
		if (abs(d) <= eta): 
			s = d * a
			over = 1
		elif d > 0: s = eta*a 
		elif d < 0: s = eta*(-a)
		elif d == 0: s = np.zeros(a.shape)

		return (s,over)

	for i in range(0,iters):
		idx = np.random.randint(0,rows)
		v = A[idx]
		s,over = step(guess, v, b[idx])
		guess = guess - s
		over_avg = gamma*over_avg + (1-gamma)*over

		if(over_avg > target_avg):
			eta = eta * step_scaling
		if(over_avg < target_avg):
			eta = eta / step_scaling
			
		error = np.linalg.norm(np.reshape(soln, cols) - guess)

		errors.append(error)

	return guess,errors


###########TESTING AND FIGURE GENERATION ###########

#Show plot and prompt to save.  Type x to avoid saving. 
def do_save(plt):
	fig = plt.gcf()
	plt.show()
	print("Save as: ", end='')
	name = input()
	if(name != "x"): fig.savefig("../../../Documents/median_presentation/" + name +".png", bbox_inches='tight')

#Runs each method, plots the l2 distance from the pseudo-solution as a function of
#the number of iterations
#type either G(gaussian) or C(corre)
def make_plots(rows, cols, errs, iters):
	A,b,soln = normalized_gaussian_with_errors(rows,cols,errs, max_error=10)
	#A,b,soln = uniform_entries_with_errors(rows, cols, low=0, high=1, errors=errs, max_error=10)

	#RK_apx, RK_errors = vanilla_RK(A, b, iters, soln)
	#plt.plot(RK_errors, label = 'RK')

	#pure_median_apx, pure_median_errors = pure_median(A, b, iters, soln, samples=100, alpha=2)
	#plt.plot(pure_median_errors, label = 'median approach')

	#SW_median_apx, SW_median_errors = RK_SW_median(A, b, iters, soln, window_size=100, alpha=2)
	#plt.plot(SW_median_errors, label = 'RK SW_median')

	#plain_SGD_apx, plain_SGD_errors = SGD_l1_min(A, b, iters, soln, eta=0.1)
	#plt.plot(plain_SGD_errors, label = 'plain SGD')

	#adap_apx, adap_errors = adaptive_l1_min(A, b, iters, soln,
	#	init_eta=1, gamma=0.99, target_avg=0.4, step_scaling=0.99)
	#plt.plot(adap_errors, label = 'adaptive SGD')

	sgd_med_apx, sgd_med_errors = SGD_median(A, b, iters, soln, samples=100)
	plt.plot(sgd_med_errors, label = 'sgd_median')

	#WARNING: Following line is slow if iters and/or rows is substantially bigger than 1000
	OPT_apx, OPT_errors = OPT_SGD(A, b, iters, soln)
	plt.plot(OPT_errors, label = 'OPT_SGD')

	plt.title("(rows,cols,errs) = ("+str(rows)+","+str(cols)+","+str(errs)+")")
	plt.legend(loc='upper right')

	do_save(plt)

def RK_vs_median_plot(rows, cols, errs, iters):

	A,b,soln = normalized_gaussian_with_errors(rows,cols,errs, max_error=10)

	pure_median_apx, pure_median_errors = pure_median(A, b, iters, soln, samples=100, alpha=1)
	RK_apx, RK_errors = vanilla_RK(A, b, iters, soln)

	thickness = 1.9
	fontsize = 16
	legendfont = 15
	plt.plot(pure_median_errors, label = 'median approach', linewidth = thickness)
	plt.plot(RK_errors, label = 'Random Kaczmarz', linewidth = thickness)

	plt.rc('font', size = fontsize)
	plt.rc('legend', fontsize = legendfont)
	plt.legend(loc='upper right')
	plt.xlabel('Iterations')
	plt.ylabel('Error')

	do_save(plt)

def median_unif_plot():
	rows,cols,errs = 50000, 100, 10000
	iters = 30000
	A,b,soln = uniform_entries_with_errors(rows, cols, low=0, high=1, errors=errs, max_error=10)

	pure_median_apx, pure_median_errors = pure_median(A, b, iters, soln, samples=100, alpha=1)
	RK_apx, RK_errors = vanilla_RK(A, b, iters, soln)


	thickness = 1.9
	fontsize = 16
	legendfont = 15
	plt.plot(pure_median_errors, label = 'median approach', linewidth = thickness)
	plt.plot(RK_errors, label = 'random Kaczmarz', linewidth = thickness)


	plt.rc('font', size = fontsize)
	plt.rc('legend', fontsize = legendfont)
	plt.legend(loc='upper right')
	plt.xlabel('Iterations')
	plt.ylabel('Error')

	do_save(plt)

def median_sgd_plot(rows, cols, errs, iters):

	A,b,soln = normalized_gaussian_with_errors(rows,cols,errs, max_error=10)
	#A,b,soln = uniform_entries_with_errors(rows, cols, low=0, high=1, errors=errs, max_error=10)


	sgd_median_apx, sgd_median_errors = SGD_median(A, b, iters, soln, samples=100)
	#RK_apx, RK_errors = vanilla_RK(A, b, iters, soln)

	thickness = 1.9
	fontsize = 16
	legendfont = 15
	plt.plot(sgd_median_errors, label = 'SGD median', linewidth = thickness)
	#plt.plot(RK_errors, label = 'random Kaczmarz', linewidth = thickness)

	plt.rc('font', size = fontsize)
	plt.rc('legend', fontsize = legendfont)
	plt.legend(loc='upper right')
	plt.xlabel('Iterations')
	plt.ylabel('Error')

	do_save(plt)


def median_plots(rows, cols, err_list, iters):

	for errs in err_list:

		#A,b,soln = normalized_gaussian_with_errors(rows,cols,errs, max_error=10)
		A,b,soln = bernoulli_with_errors(rows, cols, errs, max_error = 10)

		SW_median_apx, SW_median_errors = RK_SW_median(A, b, iters, soln, window_size=100, alpha=2)
		plt.plot(SW_median_errors, label = 'errs='+str(errs))
	plt.legend(loc='upper right')
	do_save(plt)

def quantile_plot(rows, cols, errs, iters):
	A,b,soln = normalized_gaussian_with_errors(rows,cols,errs, max_error=1)
	sgd_q1_apx, sgd_q1_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.5)

	plt.plot(sgd1_q1_errors, label = 'SGD_quantile')
	plt.legend(loc='upper right')

	do_save(plt)
	#sgd_q2_apx, sgd_q2_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.2)
	#sgd_q3_apx, sgd_q3_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)
	#sgd_q4_apx, sgd_q4_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.4)
	#sgd_q5_apx, sgd_q5_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.5)
	#sgd_q6_apx, sgd_q6_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.6)
	#sgd_q7_apx, sgd_q7_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.7)
	#sgd_q8_apx, sgd_q8_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.8)
	#sgd_q9_apx, sgd_q9_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.9)


def test_example(rows, cols, errs, iters):
	A,b,soln = normalized_gaussian_with_errors(rows,cols,15000, max_error=1)
	sgd1_median_apx, sgd1_median_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)
	A, b, soln = normalized_gaussian_with_errors(rows, cols, 15000, max_error=10)
	sgd2_median_apx, sgd2_median_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)

	A, b, soln = normalized_gaussian_with_errors(rows, cols, 15000, max_error=50)
	sgd3_median_apx, sgd3_median_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)

	A, b, soln = normalized_gaussian_with_errors(rows, cols, 15000, max_error=100)
	sgd4_median_apx, sgd4_median_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)

	A, b, soln = normalized_gaussian_with_errors(rows, cols, 15000, max_error=500)
	sgd5_median_apx, sgd5_median_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)

	A, b, soln = normalized_gaussian_with_errors(rows, cols, 15000, max_error=1000)
	sgd6_median_apx, sgd6_median_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)

	#A, b, soln = uniform_entries_with_errors(rows, cols, 0.7, 1, errs, max_error=0.5)

	#pure_med_apx, pure_med_errors = pure_median(A,b,iters,soln)
	#motzkin_apx, motzkin_errors = motzkin_hybrid(A, b, iters, soln,errs)
	#RK_apx, RK_errors = vanilla_RK(A, b, iters, soln)
	#SW_median_apx, SW_median_errors = RK_SW_median(A, b, iters, soln, window_size=100, alpha=2)
	#sgd_median_apx, sgd_median_errors = SGD_median(A, b, iters, soln, samples=100)

	#sgd_q1_apx, sgd_q1_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.1)
	#sgd_q2_apx, sgd_q2_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.2)
	#sgd_q3_apx, sgd_q3_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.3)
	#sgd_q4_apx, sgd_q4_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.4)
	#sgd_q5_apx, sgd_q5_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.5)
	#sgd_q6_apx, sgd_q6_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.6)
	#sgd_q7_apx, sgd_q7_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.7)
	#sgd_q8_apx, sgd_q8_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.8)
	sgd_q9_apx, sgd_q9_errors = SGD_quantiles(A, b, iters, soln, samples=100, quantile  = 0.9)

	#SW_q1_apx, SW_q1_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.1)
	#SW_q2_apx, SW_q2_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.2)
	#SW_q3_apx, SW_q3_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.3)
	#SW_q4_apx, SW_q4_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.4)
	#SW_median_apx, SW_median_errors = RK_SW_median(A, b, iters, soln, window_size=100, alpha=1)
	#SW_q6_apx, SW_q6_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.6)
	#SW_q7_apx, SW_q7_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.7)
	#SW_q8_apx, SW_q8_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.8)
	#SW_q9_apx, SW_q9_errors = RK_SW_quantile(A, b, iters, soln, window_size=100, quantile=0.9)

	#opt_sgd_apx, opt_sgd_errors =OPT_SGD(A, b, iters, soln)
	
	#plt.plot(motzkin_errors, label = 'motzkin-median hybrid')
	#plt.plot(RK_errors, 'b-', label = r'$\eta^{RK}$')

	#plt.plot(SW_q1_errors, label = r'$\eta^{RK\pi_{0.1}}$')
	#plt.plot(SW_q2_errors, label = r'$\eta^{RK\pi_{0.2}}$')
	#plt.plot(SW_q3_errors, label = r'$\eta^{RK\pi_{0.3}}$')
	#plt.plot(SW_q4_errors, label = r'$\eta^{RK\pi_{0.4}}$')
	#plt.plot(SW_median_errors, 'r--',label = r'$\eta^{RKm}$')
	#plt.plot(SW_q6_errors, label = r'$\eta^{RK\pi_{0.6}}$')
	#plt.plot(SW_q7_errors, label = r'$\eta^{RK\pi_{0.7}}$')
	#plt.plot(SW_q8_errors, label = r'$\eta^{RK\pi_{0.8}}$')
	#plt.plot(SW_q9_errors, label = r'$\eta^{RK\pi_{0.9}}$')
	plt.plot(sgd1_median_errors, label = 'max corruption size 1')
	plt.plot(sgd2_median_errors, label = 'max corruption size 10')

	plt.plot(sgd3_median_errors, label = 'max corruption size 50')

	plt.plot(sgd4_median_errors, label = 'max corruption size 100')

	plt.plot(sgd5_median_errors, label = 'max corruption size 500')

	plt.plot(sgd6_median_errors, label = 'max corruption size 1K')


	#plt.plot(opt_sgd_errors, 'c:', label=r'$\eta^*$')

	#plt.plot(sgd_q1_errors, label = r'$\eta^{SGD\pi_{0.1}}$')
	#plt.plot(sgd_q2_errors, label = r'$\eta^{SGD\pi_{0.2}}$')
	#plt.plot(sgd_q3_errors, label = r'$\eta^{SGD\pi_{0.3}}$')
	#plt.plot(sgd_q4_errors, label = r'$\eta^{SGD\pi_{0.4}}$')
	#plt.plot(sgd_q5_errors, label = r'$\eta^{SGD\pi_{0.5}}$')
	#plt.plot(sgd_q6_errors, label = r'$\eta^{SGD\pi_{0.6}}$')
	#plt.plot(sgd_q7_errors, label = r'$\eta^{SGD\pi_{0.7}}$')
	#plt.plot(sgd_q8_errors, label = r'$\eta^{SGD\pi_{0.8}}$')
	##plt.plot(sgd_q9_errors, label = r'$\eta^{SGD\pi_{0.9}}$')

	plt.legend(loc='upper right')
	plt.xlabel('Iterations')
	plt.ylabel('Relative error')
	#plt.show()
	plt.savefig('corruptions_sizes.png')

def adversarial_test(rows, cols, iters, frac_corrupted):
	A = normalized_gaussian_matrix(rows,cols)
	b,soln = adversarially_corrupted(A, corrupted_fraction = frac_corrupted)
	sgd_median_apx, sgd_median_errors = SGD_median(A, b, iters, soln, samples=100)

	thickness = 1.9
	fontsize = 16
	legendfont = 15
	plt.plot(sgd_median_errors, label = 'SGD median', linewidth = thickness)
	#plt.plot(RK_errors, label = 'random Kaczmarz', linewidth = thickness)

	plt.rc('font', size = fontsize)
	plt.rc('legend', fontsize = legendfont)
	plt.legend(loc='upper right')
	plt.xlabel('Iterations')
	plt.ylabel('Error')

	do_save(plt)


#test_example(rows=50000, cols=100, errs=25000, iters=3000)
#RK_vs_median_plot(50000, 100, 1000, 10000)
#make_plots(5000, 100, 500, 3000)
#median_sgd_plot(50000, 100, 15000, 20000)

#adversarial_test(50000, 100, 10000, 0.25)
