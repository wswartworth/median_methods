import numpy as np

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
def normalized_gaussian_with_errors(rows, cols, errors, *, max_error=1):
	A,b,x = normalized_gaussian(rows,cols)

	bad_rows = np.random.choice(rows, errors, replace = False)
	for i in bad_rows:
		b[i] = np.random.uniform(-max_error, max_error)
	return (A,b,x)


def uniform_entries_with_errors(rows, cols, errors, *, low=0, high=1, max_error=1):

    """
    Entries of A are initially sampled uniform in (low,high). Then each row of A is normalized.
    This can be used to generate poorly-conditioned systems.
    """
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

def bernoulli_with_errors(rows, cols, errors, *, max_error=1):
	A = np.random.binomial(1, 0.5, (rows,cols))*2 - np.ones((rows,cols));
	for i in range(0,rows):
		A[i] = A[i]/(np.linalg.norm(A[i]))
	x = np.random.uniform(-5, 5, (cols, 1))
	b = np.matmul(A, x)

	bad_rows = np.random.choice(rows, errors, replace = False)
	for i in bad_rows:
		b[i] = np.random.uniform(-max_error, max_error)
	return (A,b,x)