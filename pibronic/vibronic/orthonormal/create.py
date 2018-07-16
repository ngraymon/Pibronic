""" this module creates an orthonormal matrix used in the creation of 'artifical' vibronic models"""

# system imports
import pickle
import sys
import os

# third party imports
import numpy as np
from scipy.linalg import expm

# local imports
from .sequences import sequence_filename
from .sequences import load_sequence
from .sequences import select_pair


def create_orthonormal_matrix_lambda_close_to_identity(order, tuning_parameter):
    """returns an orthonormal matrix U which is parameterized by the tuning_parameter
    If tuning_parameter is 0.0 then it returns Identity.
    For values > 0.0 it returns U's of increasing 'distance' from Identity
    The 'distance' is defined as the matrix norm of the difference of U and Identity
    takes: the order of the matrix and tuning_parameter
    """

    # Once we finish the code to generate sequences we can re-enable this
    if True:
        assert 0.0 <= tuning_parameter <= 1.0, f"The tuning parameter ({tuning_parameter:}) is restricted to [0.0, 1.0]"

    if tuning_parameter == 0.0:
        return np.eye(order)

    upper_tri = np.tri(order, k=0).T

    skew_symmetric_matrix = upper_tri - upper_tri.T

    # Once we finish the code to generate sequences we can re-enable this
    if True:
        sequence = load_sequence(order)
        pair = select_pair(sequence, tuning_parameter)
        tuning_parameter = pair[1]

    # generate an orthonormal matrix, which depends on the tuning parameter
    mat = expm(tuning_parameter * skew_symmetric_matrix)
    assert np.allclose(mat.dot(mat.T), np.eye(order)), "matrix is not orthonormal"
    return mat


if (__name__ == "__main__"):
    assert len(sys.argv) is 3, "Need two arguments"

    order = sys.argv[1]
    tuning_parameter = sys.argv[1]

    assert isinstance(order, int), "The first argument must be the order of the matrix (type int)"
    assert isinstance(tuning_parameter, float), "The first argument must be the tuning parameter (type float)"

    create_orthonormal_matrix_lambda_close_to_identity(order, tuning_parameter)
