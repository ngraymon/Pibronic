""" this module creates an orthonormal matrix used in the creation of 'artificial' vibronic models"""

# system imports
import sys

# third party imports
import numpy as np
from scipy.linalg import expm

# local imports
from . import sequences as seq


def create_orthonormal_matrix_lambda_close_to_identity(order, tuning_parameter):
    """returns an orthonormal matrix U which is parameterized by the tuning_parameter
    If tuning_parameter is 0.0 then it returns Identity.
    For values > 0.0 it returns U's of increasing 'distance' from Identity
    The 'distance' is defined as the matrix norm of the difference of U and Identity
    takes: the order of the matrix and tuning_parameter
    """

    assert 0.0 <= tuning_parameter <= 1.0, f"The tuning parameter ({tuning_parameter:}) is restricted to [0.0, 1.0]"

    SKSM, identity = seq.build_SKSM_and_identity(order)

    # a tuning parameter of 0 should map to identity
    if tuning_parameter == 0.0:
        return identity

    sequence = seq.load_sequence(order)
    pair = seq.select_pair(sequence, tuning_parameter)

    # generate the orthonormal matrix
    mat = expm(pair.x * SKSM)

    assert np.allclose(mat.dot(mat.T), np.eye(order)), "matrix is not orthonormal"
    return mat


def main(args):
    assert len(args) is 3, "Need two arguments"

    order = args[1]
    tuning_parameter = args[2]

    assert isinstance(order, int), "The first argument must be the order of the matrix (type int)"
    assert isinstance(tuning_parameter, float), "The second argument must be the tuning parameter (type float)"

    create_orthonormal_matrix_lambda_close_to_identity(order, tuning_parameter)
    return


if (__name__ == "__main__"):
    main(sys.argv)
