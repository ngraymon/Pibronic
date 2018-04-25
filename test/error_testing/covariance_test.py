# a simple independent script to show that the method for calculating the covariance is correct

from .context import pibronic
from numpy.linalg import LinAlgError
import numpy as np
import logging
import timeit

# set a format which is simpler for console use, can be used for file as well if needed
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# tell the handler to use this format
#console.setFormatter(formatter)

file = logging.FileHandler(filename='debug.log', mode='w')
file.setLevel(logging.DEBUG)
#file.setFormatter(formatter) # optional

# add the handler to the root logger
oak = logging.getLogger('test_sampling')
oak.setLevel(logging.DEBUG)
oak.addHandler(console)
oak.addHandler(file)

debugging = True

#   print("Diag version", timeit.timeit(stmt=s1, number=1000000), "Loop version", timeit.timeit(stmt=s2, number=1000000), sep=" ")

# checking for Positive semi-definite
from scipy.sparse.linalg import arpack
def isPositiveSemiDefinite(A, tol = 1e-8):
    vals, vecs = arpack.eigsh(A, k=2, which ='BE') # return the ends of spectrum of A
    return np.all(vals > -tol)

# makes life easier atm
boltzman_constant = 8.6173324E-5 #(eV/K
temperature = 300.0 #K
hbar = 1.0
beta = 1.0 / (boltzman_constant * temperature)
#oak.info("Beta = {}".format(beta))
size = 10

number_of_electronic_surfaces   = 3
number_of_normal_modes          = 3
number_of_beads                 = 5


surface_shift_array = np.arange(0.1, 0.1*number_of_electronic_surfaces, 0.1)
omega_array = np.arange(0.025, 0.025*number_of_normal_modes, 0.025)
#oak.info("Omega values: {}".format(omega_array))
#omega_array = np.array([0.1]*number_of_normal_modes)

surface_array = np.array(range(number_of_electronic_surfaces))
mode_array = np.array(range(number_of_normal_modes))

# we'll index it by surface, then mode, then bead
#input_data_array = np.zeros((number_of_electronic_surfaces, number_of_normal_modes, number_of_beads))
# identity?
#input_data_array = np.eye(row, column)


# pick our seed
#np.random.seed(35)


np.set_printoptions(precision=8, linewidth=300)
# randomly pick a surface
surface_sample_drawn_from = np.random.randint(0, number_of_electronic_surfaces)

# create the O matrix
O_matrix = np.zeros((number_of_beads, number_of_beads))
for a in range(0, number_of_beads-1):
        O_matrix[a, a+1] = 1
        O_matrix[a+1, a] = 1
# add the corners to the array
O_matrix[0,-1] +=1
O_matrix[-1,0] +=1 
oak.debug("O_matrix\n{}".format(O_matrix))

# diagonalize O
O_eigenvals, O_eigenvectors = np.linalg.eigh(O_matrix) # note that we can specify which part of the triangle to use
oak.debug("O_eigenvals{}\nO_eigenvectors\n{}".format(O_eigenvals, O_eigenvectors))

# calculate all at once
co_tangent_matrix = np.power(np.tanh(omega_array*hbar*beta), -1)
co_secant_matrix = np.power(np.sinh(omega_array*hbar*beta), -1)

oak.debug("co_tangent_matrix: {}\nco_secant_matrix: {}".format(co_tangent_matrix, co_secant_matrix))

covariance_matrix = np.subtract(2 * co_tangent_matrix[0] * np.identity(number_of_beads), np.diagflat(co_secant_matrix[0] * O_eigenvals, k=0))
#covariance_matrix = np.identity(number_of_beads, dtype=np.float64)
oak.info("Covariance matrix\n{}".format(covariance_matrix))
number_of_samples = np.int64(1e6)
oak.info("Taking {} samples".format(number_of_samples))

q2_average_array = np.zeros(number_of_beads)
q_average_array = np.zeros(number_of_beads)
average_array = np.zeros(number_of_beads)

# (samples, beads)
collective_coordinate_sample = np.random.multivariate_normal(np.zeros(number_of_beads), np.linalg.inv(covariance_matrix), number_of_samples)
displaced_coordinates = np.einsum('ij,...j->...i', O_eigenvectors.T, np.einsum('ij,...j->...i', O_eigenvectors, collective_coordinate_sample, dtype=np.float64) + np.float64(0.1), dtype=np.float64)
if(False):
	oak.info("<q^2>: {}".format(np.mean(np.power(collective_coordinate_sample, 2), axis=0, dtype=np.float64)))
	oak.info("<(q+d)^2> - <q+d>^2: {}".format(np.mean(np.power(displaced_coordinates, 2), axis=0, dtype=np.float64) - np.power(np.mean(displaced_coordinates, axis=0, dtype=np.float64), 2)))
else:
	# check
    # np.sum(np.inner(displaced_coordinates[0], np.multiply(covariance_matrix, displaced_coordinates[0])))
    q_alpha_q = np.einsum('...i,...i', displaced_coordinates, np.einsum('ij,...j->...i', covariance_matrix, displaced_coordinates))
    q_alpha_q = np.multiply(np.float64(-0.5), q_alpha_q)
    exp_q_alpha_q = np.exp(q_alpha_q)
    oak.info("Number of samples {}\nNumber of positive samples {}\nNumber of negative samples {}\nNumber of zero exponents {}".format(number_of_samples, q_alpha_q[q_alpha_q > 0.0].size, q_alpha_q[q_alpha_q < 0.0].size, exp_q_alpha_q[np.isclose(exp_q_alpha_q, 0.0)].size))



