""" this module handles sequences of (y, x) pairs which are used in the creation of orthonormal matricies,

The orthonormal matricies are used to create 'artificial' vibronic models (S):
 - To begin you choose any real square matrix (E), which we can consider the 'eigenvalues' of the desired output matrix (S).
 - Next a orthonormal matrix (U) is generated, we can consider these the 'eigenvectors' of the desired output matrix (S).
 - Finally we preform a unitary transformation on (E) using (U) which gives us (S).

Of importance is how (U) is generated.
It would be optimal to have a function g(A, P) which genereates an orthonormal matrix (U).
The parameter (A) would define the order of (U).
The 'tuning' parameter (P) would define the 'distance' of (U) from the identity matrix of the same order.
The function g's domain, for a fixed (A), would be the reals in [0.0, 1.0] and the range would be a matrix in [Identity, M], where (M) is the 'farthest' matrix from Identity.
Thus as we increase the value of (P), for a fixed (A), the trace of (U) should decrease and the off-diagonal values should increase.
A challenging problem is defining the meaning of 'distance' for an arbitrary square matrix?
We chose to define the 'distance' as the Frobenius norm (AKA matrix norm) of the difference between (U) and Identity:
'distance(P)' = ||U(P) - Identity||

Our first attempt at solving this problem was:
 - Generate a random matrix (K) of order A
 - Create a skew symmetric matrix S = K - K.T
 - Calculate the orthonormal matrix U(P) = expm(P * S)

However because we randomly generated the matrix (K) this method suffers from periodicity issues.
For a matrix of order 2, the 'distance' of (U) from Identity is periodic, and the period depends on the matrix (K).
It behaves as follows:
 - P = 0            : U is Identity
 - P = period / 4   : U is maximally diagonal and has a trace of 0
 - P = period / 2   : U is -Identity and has a trace of -2
 - P = 3 period / 4 : U is maximally diagonal and has a trace of 0

For a matrix of order 3 or higher it is non trivial to determine the period



Therefore we have chosen to go with a repeatable solution:
Instead of randomly generating the matrix (K) we always choose (K) to be a upper unitriangular matrix of order (A).
This allows us to precalculate an estimate of the period of the 'distance' of the matrix (U).
Our approach to finding the period, for a fixed (A), works as follows:
 - Generate an upper unitriangular matrix (K) of order A
 - Create a skew symmetric matrix S = K - K.T
 - Calculate the 'distance(x)' = ||U(x) - Identity|| for x in [0, 1E6]
 - Plot y='distance(x)' and visually or algorithmically select a continous sequence of increasing y values, optimally the sequence begins very close to identity
 - This sequence of pairs [(y_{i}, x_{i}) (y_{i+n}, y_{i+n}] is then saved to a file

For each choice of (A) we can then obtain a sequence which forms the range of the function g(A, P).
We restrict the domain of (P) values to be [0.0, 1.0] and choose a mapping of any P > 0.0 to a given pair in the sequence (y, x).

Thus the code for creating an orthonormal matrix which is used to create an 'artificial' vibronic models (S) is:
 - Generate an upper unitriangular matrix (K) of order A
 - Create a skew symmetric matrix S = K - K.T
 - If P = 0.0 then U = Identity, otherwise:
    - Retrive a pair (y,x) using the tuning parameter (P)
    - Calculate the orthonormal matrix U = expm(x * S)
"""

# system imports
import pickle
# import sys
import os

# third party imports
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

# local imports

sequence_filename = "sequence_{:d}.data"
minimum_length_of_sequence = 50  # this is a guess at the moment


def load_sequence(order):
    """ returns a list of [y, x] pairs (which are length 2 lists)
    throws an assert error if the file does not exist
    takes a integer parameter which specifies which file to load"""

    # generate the path to create.py's directory
    dir_root = os.path.dirname(os.path.realpath(__file__))

    path = os.path.join(dir_root, sequence_filename.format(order))

    assert os.path.isfile(path), f"File {path:s} not found! "

    sequence = np.loadtxt(path, delimiter=', ')

    return sequence.tolist()


def save_sequence(order, sequence):
    """ saves a list of [y, x] pairs (which are length 2 lists)
    pickles them to a file named using the string sequence_filename, in the directory that
    this module is located in"""

    assert len(sequence) > minimum_length_of_sequence, f"There are less than {minimum_length_of_sequence:} items in the sequence"

    # generate the path to create.py's directory
    dir_root = os.path.dirname(os.path.realpath(__file__))

    path = os.path.join(dir_root, sequence_filename.format(order))

    np.savetxt(path, sequence, fmt="%.18e", delimiter=', ')

    return


def select_pair(sequence, tuning_parameter):
    """ holds the logic for selecting a (y, x) pair from the sequence
    that matches the tuning parameter"""

    # TODO - how to preform this mapping needs to be determined

    return (0.0, 0.0)


def find_first_turn_around_point(SKSM, identity):
    """ return a tuple (y, x) which are the values before the first turn around point """

    old_distance = None
    old_x = None

    period = 5e2

    for x in np.linspace(0, period, 1e6):
        distance = norm(expm(x * SKSM) - identity)
        if old_distance is not None and old_distance > distance:
            print("we found the turn around point at from values\n"
                  "({old_distance:},{old_x})\nto\n({distance},{x})")
            break
        old_distance = distance
        old_x = x

    else:
        assert False, "oh no things went wrong, everything is on fire!!!!"

    return (old_distance, old_x)


def generate_sequence(order):
    """ x """

    # we always start with an upper unitriangular matrix
    upper_tri = np.tri(order, k=0).T

    # which we use to generate the skew_symmetric_matrix
    skew_symmetric_matrix = upper_tri - upper_tri.T

    # if we diagonalize the skew_symmetric_matrix can we re-cacluate the matrix exponential easily?
    # it seems we can't treat it as a symmetric matrix? they numpy methods use either the upper or lower triangle, which would be incorrect

    identity = np.eye(order)

    # TODO - this will probably be refined in the future
    sequence = [(norm(expm(x * skew_symmetric_matrix) - identity), x) for x in np.linspace(0, 1e2, 1e4)]

    save_sequence(order, sequence)

    return


if (__name__ == "__main__"):
    for order in range(10):
        generate_sequence(order)
