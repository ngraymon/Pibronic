""" this module handles sequences of (y, x) pairs which are used in the creation of orthonormal matrices,

# TODO - this section is out of date and should probably be modified

The orthonormal matrices are used to create 'artificial' vibronic models (S):

*  To begin you choose any real square matrix (E), which we can consider the 'eigenvalues' of the desired output matrix (S).
*  Next a orthonormal matrix (U) is generated, we can consider these the 'eigenvectors' of the desired output matrix (S).
*  Finally we preform a unitary transformation on (E) using (U) which gives us (S).

Of importance is how (U) is generated.
It would be optimal to have a function g(A, P) which generates an orthonormal matrix (U).
The parameter (A) would define the order of (U).
The 'tuning' parameter (P) would define the 'distance' of (U) from the identity matrix of the same order.
The function g's domain, for a fixed (A), would be the reals in [0.0, 1.0] and the range would be a matrix in [Identity, M], where (M) is the 'farthest' matrix from Identity.
Thus as we increase the value of (P), for a fixed (A), the trace of (U) should decrease and the off-diagonal values should increase.
A challenging problem is defining the meaning of 'distance' for an arbitrary square matrix?
We chose to define the 'distance' as the Frobenius norm (AKA matrix norm) of the difference between (U) and Identity:
'distance(P)' = ||U(P) - Identity||

Our first attempt at solving this problem was:
*  Generate a random matrix (K) of order A
*  Create a skew symmetric matrix S = K - K.T
*  Calculate the orthonormal matrix U(P) = expm(P * S)

However because we randomly generated the matrix (K) this method suffers from periodicity issues.
For a matrix of order 2, the 'distance' of (U) from Identity is periodic, and the period depends on the matrix (K).
It behaves as follows:

*  P = 0            : U is Identity
*  P = period / 4   : U is maximally diagonal and has a trace of 0
*  P = period / 2   : U is -Identity and has a trace of -2
*  P = 3 period / 4 : U is maximally diagonal and has a trace of 0

For a matrix of order 3 or higher it is non trivial to determine the period



Therefore we have chosen to go with a repeatable solution:
Instead of randomly generating the matrix (K) we always choose (K) to be a upper unitriangular matrix of order (A).
This allows us to precalculate an estimate of the period of the 'distance' of the matrix (U).
Our approach to finding the period, for a fixed (A), works as follows:

*  Generate an upper unitriangular matrix (K) of order A
*  Create a skew symmetric matrix S = K - K.T
*  Calculate the 'distance(x)' = ||U(x) - Identity|| for x in [0, 1E6]
*  Plot y='distance(x)' and visually or algorithmically select a continuous sequence of increasing y values, optimally the sequence begins very close to identity
*  This sequence of pairs [(y_{i}, x_{i}) (y_{i+n}, y_{i+n}] is then saved to a file

For each choice of (A) we can then obtain a sequence which forms the range of the function g(A, P).
We restrict the domain of (P) values to be [0.0, 1.0] and choose a mapping of any P > 0.0 to a given pair in the sequence (y, x).

Thus the code for creating an orthonormal matrix which is used to create an 'artificial' vibronic models (S) is:

*  Generate an upper unitriangular matrix (K) of order A
*  Create a skew symmetric matrix S = K - K.T
*  If P = 0.0 then U = Identity, otherwise:

   *  Retrieve a pair (y,x) using the tuning parameter (P)
   *  Calculate the orthonormal matrix U = expm(x * S)
"""

# system imports
from collections import namedtuple
from functools import partial
from os.path import dirname, join, realpath, isfile, exists

# third party imports
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

# local imports


sequence_filename = "sequence_{:d}.data"
x_pair_filename = "sequence_{:d}.pairs"
minimum_length_of_sequence = 50  # this is a guess at the moment

# all functions share the same constants and therefore we can load sequences reliably
n_bins = int(1e4)
max_value = int(1e2)
linspace_dims = (0, max_value, n_bins)

# use this for passing around x,y pairs
Pair = namedtuple('pair', ['x', 'y'])


def difference_function(skew_symmetric_matrix, identity):
    """ returns a function which calculates the difference between the unitary matrix, parameterized by alpha, and identity """
    return lambda x: expm(x * skew_symmetric_matrix) - identity


def check_minimum_length_of_sequence(sequence):
    """ asserts that a sequence's length is > the defined minimum length"""
    error_string = f"There are less than {minimum_length_of_sequence:} items in the sequence"
    assert len(sequence) >= minimum_length_of_sequence, error_string
    return


def path_to_x_pair_file(order):
    """ returns the path to a x_pair file associated with the order argument"""
    dir_root = dirname(realpath(__file__))
    assert exists(dir_root), f"module path {dir_root:s} does not exist!?!?"
    path = join(dir_root, x_pair_filename.format(order))
    return path


def path_to_sequence_file(order):
    """ returns the path to a sequence file associated with the order argument"""
    dir_root = dirname(realpath(__file__))
    assert exists(dir_root), f"module path {dir_root:s} does not exist!?!?"
    path = join(dir_root, sequence_filename.format(order))
    return path


def load_sequence(order, delimiter=", "):
    """ returns a list of [y, x] pairs (which are length 2 lists)
    throws an assert error if the file does not exist
    takes a integer parameter which specifies which file to load"""
    path = path_to_sequence_file(order)
    assert isfile(path), f"Sequence {path:s} is not a valid file!?"
    sequence = np.loadtxt(path, delimiter=delimiter)
    return sequence.tolist()


def save_sequence(order, sequence, fmt="%.18e", delimiter=", "):
    """ saves a list of iterables, commonly they are length 2 and contain floats, but the fmt parameter can be changed for flexibility
    pickles them to a file named using the string sequence_filename, in the directory that
    this module is located in"""
    check_minimum_length_of_sequence(sequence)
    np.savetxt(path_to_sequence_file(order), sequence, fmt=fmt, delimiter=delimiter)
    return


def select_pair(sequence, tuning_parameter):
    """ holds the logic for selecting a (y, x) pair from the sequence
    that matches the tuning parameter"""
    assert 0.0 <= tuning_parameter <= 1.0, "The tuning parameter is restricted to [0.0, 1.0]"

    max_y_value = sequence[-1][0]
    desired_y_value = max_y_value * tuning_parameter

    if tuning_parameter == 0.0:
        return Pair(x=0.0, y=0.0)

    if tuning_parameter == 1.0:
        return Pair(x=sequence[-1][1], y=sequence[-1][0])

    for idx, s in enumerate(sequence):
        if np.isclose(s[0], desired_y_value):
            return Pair(x=s[1], y=s[0])
        # if the current 'pair' has a larger y value, select the last pair
        if s[0] > desired_y_value:
            return Pair(x=sequence[idx-1][1], y=sequence[idx-1][0])

    raise Exception("We couldn't find an appropriate pair!!")


def build_SKSM_and_identity(order):
    """ returns the skew symmetric matrix and identity matrix for the given matrix order
    this is code that was factored out of other functions """

    assert isinstance(order, int) and order > 0, "order must be integer value greater than 0"

    # we choose to always use an upper unitriangular matrix
    upper_tri = np.tri(order, k=0).T
    skew_symmetric_matrix = upper_tri - upper_tri.T
    identity = np.eye(order)

    return skew_symmetric_matrix, identity


def find_first_turn_around_point(SKSM, identity):
    """ return a tuple (y, x) which are the values before the first turn around point """

    old_distance = None
    old_x = None

    for x in np.linspace(*linspace_dims):
        distance = norm(expm(x * SKSM) - identity)
        if old_distance is not None and old_distance > distance:
            print("we found the turn around point from values\n"
                  f"({old_distance:},{old_x})\nto\n({distance},{x})")
            break
        old_distance = distance
        old_x = x

    else:
        assert False, "oh no things went wrong, everything is on fire!!!!"

    return (old_distance, old_x)


def generate_monotonic_sequence_to_first_turn_around(order):
    """ generates a sequence of the frobenius norm to the first turn around point
    which will be used when generating new matrices"""
    SKSM, identity = build_SKSM_and_identity(order)

    max_distance, max_x = find_first_turn_around_point(SKSM, identity)
    # the first turn around point is when the matrix wraps back around to -Identity so we want half that distance

    end_point = max_x / 2.0

    diff = difference_function(SKSM, identity)
    frobenius_norm = partial(norm, ord='fro')
    sequence = [(frobenius_norm(diff(x)), x) for x in np.linspace(0, end_point, n_bins)]

    save_sequence(order, sequence)
    return


def generate_and_save_x_pairs(order, p_list):
    """ generates and saves a list of tuples (containing 2 floats)
    the 2 floats are lower and upper bounds on acceptable x values to use in generating a monotonic sequence
    saves them to a file named using the string x_pair_filename, in the directory that
    this module is located in"""
    x_pairs = [(0., p_list[0].x), ]
    for idx in range(1, len(p_list), 2):
        x_pairs.append((p_list[idx].x, p_list[idx+1].x))

    np.savetxt(path_to_x_pair_file(order), x_pairs, fmt="%.18e", delimiter=", ")
    return x_pairs


def list_of_turning_points(SKSM, identity):
    """ return a list of tuples (y, x) which are the values before a turning point
    these are the last points before the function begins decreasing """

    descending = False
    previous_turn_around = None
    old_distance = 0.0
    old_x = 0.0

    xy_list = []

    diff = difference_function(SKSM, identity)
    frobenius_norm = partial(norm, ord='fro')

    for x in np.linspace(*linspace_dims):
        distance = frobenius_norm(diff(x))

        if not descending and old_distance > distance:
            xy_list.append(Pair(x=old_x, y=old_distance))
            previous_turn_around = old_distance
            descending = True

        elif descending and previous_turn_around <= distance:
            xy_list.append(Pair(x=x, y=distance))
            descending = False

        old_distance = distance
        old_x = x

    return xy_list


def generate_full_monotonic_sequence(order):
    """ saves two files, sequence_#.data and sequence_#.pairs
    sequence_#.pairs are the x pairs representing sequences where the frobenius norm is monotonic
    sequence_#.data is a list of x, y values
    this sequence is discontinuous and built up of x"""
    SKSM, identity = build_SKSM_and_identity(order)

    # the list of turning points allows us to select increasing x,y pairs
    p_list = list_of_turning_points(SKSM, identity)

    # we generate the x pairs and save them to a file
    x_pairs = generate_and_save_x_pairs(order, p_list)

    # the period is from (0, max_x) and (0, max_y)
    numerical_max_y = p_list[-1].y
    numerical_max_x = p_list[-1].x

    # the analytical value for the maximum y is 2 * sqrt(order)
    analytical_max_y = 2 * np.sqrt(order)
    analytical_max_x = None  # we can't know this value ahead of time!

    # we choose to use the analytical option as our maximum y value
    analytical = True
    max_y = analytical_max_y if analytical else p_list[-1].y

    # the tuning parameter which gives the 'maximally entangling' U matrix is 1/4 of the period
    best_y = max_y / 4.

    diff = difference_function(SKSM, identity)
    frobenius_norm = partial(norm, ord='fro')

    # we find the closest x value to get us that 1/4 of the period
    samples, step = np.linspace(0., numerical_max_x, n_bins, retstep=True)
    for x in samples:
        y = frobenius_norm(diff(x))
        if y > best_y:
            best_x = x - step  # just to make sure we don't overshoot
            new_best_y = frobenius_norm(diff(best_x))
            assert new_best_y <= best_y, "Our new y is larger than the y we wanted to find?!"
            break
    else:
        raise Exception("All of the y values were less than 1/4 the largest y value?!?")

    # first we calculate the full sequence
    full_sequence = [(frobenius_norm(diff(x)), x) for x in np.linspace(0., best_x, n_bins)]

    # then we prune off the data points that are between the discontinuities in our sequence
    pruned_sequence = []
    for x_pair in x_pairs:
        pruned_sequence.extend(filter(lambda p: x_pair[0] <= p[1] <= x_pair[1], full_sequence))

    """ testing has shown that the sequences are always inside the first monotonic sequence of the matrix's period as indicated by the pruned sequence having the same length as the full_sequence, none of the elements in the full_sequence are in a region of decreasing value
    """
    print(len(full_sequence), len(pruned_sequence))

    save_sequence(order, pruned_sequence)
    return


def generate_full_monotonic_sequence_for_plotting(order):
    """ generates one data file for plotting with gnuplot """

    SKSM, identity = build_SKSM_and_identity(order)

    p_list = list_of_turning_points(SKSM, identity)

    # first we generate the x pairs and save them to a file
    x_pairs = generate_and_save_x_pairs(p_list)
    max_x = p_list[-1].x

    diff = difference_function(SKSM, identity)
    frobenius_norm = partial(norm, ord='fro')

    # first we calculate the full sequence
    full_sequence = [(frobenius_norm(diff(x)), x) for x in np.linspace(0., max_x, n_bins)]

    # then we prune off the data points that are between the discontinuities in our sequence
    pruned_sequence = []
    for x_pair in x_pairs:
        pruned_sequence.extend(filter(lambda p: x_pair[0] <= p[1] <= x_pair[1], full_sequence))

    # we create a sequence with None's where we don't want to plot points
    pruned_y_values = []
    for idx, p in enumerate(full_sequence):
        if p not in pruned_sequence:
            pruned_y_values.append(None)
        else:
            pruned_y_values.append(p[0])

    sequence = [(
                frobenius_norm(diff(x)),
                pruned_y_values[idx],
                x,
                ) for idx, x in enumerate(np.linspace(0., max_x, n_bins))]

    save_sequence(order, sequence, fmt="%.18e, %s, %.18e")
    return


def generate_turning_point_sequence_for_plotting(order):
    """ this generates two sequences, the frobenius norm and the 'turning points' for plotting to visually confirm that those data points can be used to generate a monotonic sequence of x,y pairs which can be used to generate increasingly coupled matrices"""

    SKSM, identity = build_SKSM_and_identity(order)

    p_list = list_of_turning_points(SKSM, identity)
    turning_points = []

    for x in np.linspace(*linspace_dims):
        if len(p_list) > 0 and np.isclose(x, p_list[0].x):
            turning_points.append(p_list[0].y)
            del p_list[0]
        turning_points.append(None)

    diff = difference_function(SKSM, identity)
    frobenius_norm = partial(norm, ord='fro')

    sequence = [(
                frobenius_norm(diff(x)),
                turning_points[idx],
                x,
                ) for idx, x in enumerate(np.linspace(*linspace_dims))]

    save_sequence(order, sequence, fmt="%.18e, %s, %.18e")
    return


def generate_sequences_for_plotting(order):
    """ this generates sequences of the 1,2 and infinity norm's of the difference between identity and the skew symmetric matrix
    this provides data that can be plotted to analyze the periodic behaviour of the norm's for different size of matrices"""

    SKSM, identity = build_SKSM_and_identity(order)

    # if we diagonalize the skew_symmetric_matrix can we re-calculate the matrix exponential easily?
    # it seems we can't treat it as a symmetric matrix? they numpy methods use either the upper or lower triangle, which would be incorrect

    # the three types of norms that we calculate
    one_norm = partial(norm, ord=1)
    frobenius_norm = partial(norm, ord='fro')
    infinity_norm = partial(norm, ord=np.inf)

    diff = difference_function(SKSM, identity)

    sequence = [(
                one_norm(diff(x)),
                frobenius_norm(diff(x)),
                infinity_norm(diff(x)),
                x,
                ) for x in np.linspace(*linspace_dims)]

    save_sequence(order, sequence, fmt="%.18e, %.18e, %.18e, %.18e")

    return


def debugging_matrix_checking_function(args):
    """ for quick checking of the unitary transformation matrix which is generated using the sequence over a sample of lambda values"""
    import sys
    order = int(sys.argv[1])
    SKSM, identity = build_SKSM_and_identity(order)
    sequence = load_sequence(order)
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pair = select_pair(sequence, x)
        mat = expm(pair.x * SKSM)
        np.set_printoptions(linewidth=200)
        print(f"The tuning parameter ({x:}) gives ", pair, "and mat\n", mat)
    return


def main():
    for order in range(2, 11, 1):
        generate_full_monotonic_sequence(order)
        print(f"Finished generating a sequence of order {order:}")


if (__name__ == "__main__"):
    main()
    # debugging_matrix_checking_function()
