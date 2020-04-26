import system_generation as sysgen
import median_methods as methods 
import matplotlib.pyplot as plt
import numpy as np

def errors_by_iteration(method, iters, soln):

	errors = []
	for i in range(0,iters):
		errors.append(method.distanceTo(soln))
		method.do_iteration()
	return errors

def make_plot(methods, iters, soln):

	for method,label in methods:
		errs = errors_by_iteration(method, iters, soln)
		plt.plot(errs, label=label)
	plt.legend()
	return plt

def plot_test():

	rows, cols, errs, iters = 10, 5, 0, 10
	A,b,soln = sysgen.normalized_gaussian_with_errors(rows,cols,errs, max_error=1)
	initvect = np.zeros(cols)

	start_data = [A,b,initvect]

	rk = methods.RK(*start_data)
	sw_rk = methods.SWQuantileRK(*start_data, quantile=0.9, window_size=100)
	sample_rk = methods.SampledQuantileRK(*start_data, quantile=0.9, samples=100)
	sw_sgd = methods.SW_SGD(*start_data, quantile=0.5, window_size=100)
	fixed_sgd = methods.FixedStepSGD(*start_data,eta=0.1)
	opt_sgd = methods.OptSGD(A,b,initvect, soln=soln)

	method_list = [[rk, "rk"], [sw_rk,"sw_rk"], [sample_rk, "sample_rk"], 
	[sw_sgd, "sw_sgd"], [fixed_sgd, "fixed_sgd"], [opt_sgd, "opt_sgd"]]

	make_plot(method_list, iters, soln).show()

plot_test()







