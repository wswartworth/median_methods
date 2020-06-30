import system_generation as sysgen
import median_methods as methods
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

from scipy.optimize import minimize_scalar


def errors_by_iteration(method, iters, soln):
    errors = []
    for i in range(0, iters):
        errors.append(method.distanceTo(soln))
        method.do_iteration()
    return errors


def final_error(method, iters, soln):
    for i in range(0, iters):
        method.do_iteration()
    return method.distanceTo(soln)


def make_plot(methods, iters, soln, *, file_name=None, linedesigns=None):
    lineind = 0
    for method, label in methods:
        errs = errors_by_iteration(method, iters, soln)
        if linedesigns is not None:
            linedesign = linedesigns[lineind]
            fig = plt.plot(errs, linedesign[0], linewidth=linedesign[1], markersize=linedesign[2],
                           markevery=linedesign[3], label=label)
            lineind = lineind + 1
        else:
            fig = plt.plot(errs, label=label)
    plt.legend()

    if file_name is not None:
        plt.savefig(file_name + '.png')
        pickle.dump(fig, open(file_name + '.pickle', 'wb'))
    return plt


'''
def quantileOptimization():

	rows, cols, errs, iters = 50000, 100, 20000, 10000
	A,b,soln = sysgen.normalized_gaussian_with_errors(rows, cols, errs, max_error=1)
	start = np.zeros(cols)
	start_data = [A,b,start]

	def obj(q):
		print(q)
		sw_sgd = methods.SW_SGD(*start_data, quantile=q, window_size=100)
		error = errors_by_iteration(sw_sgd,iters,soln)[-1]
		return error

	res = minimize_scalar(obj, bounds=(0, 1), method='bounded', options={'maxiter':20})
	print("opt quantile: ", res.x)
'''


def sgd_various_quantiles(start_data, soln, iters):
    method_list = [methods.SampledQuantileSGD(*start_data,
                                              quantile=q / 10, samples=100) for q in range(1, 9)]

    return make_plot(method_list, iters, soln).show()


def make_plots():
    rows, cols = 50000, 100
    beta = 0.2
    errs = math.ceil(beta * rows);
    A, b, soln = sysgen.normalized_gaussian_with_errors(rows, cols, errs, max_error=1)
    start = np.zeros(cols)
    start_data = [A, b, start]

    sgd_various_quantiles = [[methods.SampledQuantileSGD(*start_data,
                                                         quantile=q / 10, samples=100), "Q=" + str(q)] for q in
                             range(1, 10)]

    rk_various_quantiles = [[methods.SampledQuantileRK(*start_data,
                                                       quantile=q / 10, samples=100), "Q=" + str(q)] for q in
                            range(1, 10)]

    rk_vs_sgd = [[methods.SampledQuantileSGD(*start_data,
                                             quantile=0.5, samples=100), "SGD"],
                 [methods.SampledQuantileRK(*start_data,
                                            quantile=0.5, samples=100), "RK"]]

    make_plot(sgd_various_quantiles, iters=2000, soln=soln).show()
    make_plot(rk_various_quantiles, iters=2000, soln=soln).show()
    make_plot(rk_vs_sgd, iters=4000, soln=soln).show()


def plot_test():
    A, b, soln = sysgen.normalized_gaussian_with_errors(rows, cols, errs, max_error=1)
    start = np.zeros(cols)
    start_data = [A, b, start]

    rk = methods.RK(*start_data)
    sw_rk = methods.SWQuantileRK(*start_data, quantile=0.9, window_size=100)
    sample_rk = methods.SampledQuantileRK(*start_data, quantile=0.9, samples=100)
    sample_sgd = methods.SampledQuantileSGD(*start_data, quantile=0.3, samples=100)
    sw_sgd = methods.SW_SGD(*start_data, quantile=0.5, window_size=100)
    fixed_sgd = methods.FixedStepSGD(*start_data, eta=0.1)
    opt_sgd = methods.OptSGD(*start_data, soln=soln)

    method_list = [
        [rk, "rk"],
        [sw_rk, "sw_rk"],
        [sample_rk, "sample_rk"],
        [sample_sgd, "sample_sgd"],
        [sw_sgd, "sw_sgd"],
        [fixed_sgd, "fixed_sgd"],
        # [opt_sgd, "opt_sgd"]
    ]

    make_plot(method_list, iters, soln).show()


def adversarial_sgd_plot():
    rows, cols = 50000, 100
    beta = 0.40
    iters = 50000
    errs = math.ceil(beta * rows)

    A = sysgen.normalized_gaussian_matrix(rows, cols)
    b, soln = sysgen.adversarially_corrupted(A, corrupted_fraction=beta)

    # A,b,soln = sysgen.normalized_gaussian_with_errors(rows, cols, errs, max_error=1)

    start = np.zeros(cols)
    start_data = [A, b, start]

    sgd = methods.SampledQuantileSGD(*start_data, quantile=0.2, samples=100)

    method_list = [
        [sgd, "sgd_0.3"]
    ]

    make_plot(method_list, iters, soln).show()
