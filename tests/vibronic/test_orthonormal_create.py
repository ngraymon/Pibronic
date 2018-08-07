""" """

# system imports

# local imports
from .. import context
from pibronic.vibronic.orthonormal import create

# third party imports
import pytest
from pytest import raises


class TestMain():

    def test_wrong_num_args(self):
        with raises(AssertionError) as e_info:
            create.main([])
        assert str(e_info.value) == "Need two arguments"

    def test_first_arg_not_int(self):
        with raises(AssertionError) as e_info:
            create.main([0, "fake", 1.0])
        assert str(e_info.value) == "The first argument must be the order of the matrix (type int)"

    def test_second_arg_not_float(self):
        with raises(AssertionError) as e_info:
            create.main([0, 2, 3])
        assert str(e_info.value) == "The second argument must be the tuning parameter (type float)"


def test_tuning_parameter():
    tp = 2.0
    with raises(AssertionError) as e_info:
        create.create_orthonormal_matrix_lambda_close_to_identity(2, tp)
    assert str(e_info.value) == f"The tuning parameter ({tp:}) is restricted to [0.0, 1.0]"
