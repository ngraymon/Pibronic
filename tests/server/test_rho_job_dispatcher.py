""" """

# system imports

# local imports
from ..context import pibronic
import pibronic.server.rho_job_dispatcher as rjd

# third party imports
import pytest


def test_estimate_memory_usuage():
    """x"""
    A = 4
    N = 2
    B = 80

    # try to use less memory than available on a single node
    assert rjd.estimate_memory_usuage(A, N, B) == 52

    # try to use more memory than available on a single node
    with pytest.raises(AssertionError, message="Asked for 8589934592GB"):
        rjd.estimate_memory_usuage(8, 4, B)

    return


template_param_dict = {
    "basis_size": 20,
    "wait_arg": "",
    "cores_requested": "12",
    # "root_dir": FS.path_data,
    # "id_data": FS.id_data,
}


@pytest.fixture(params=[key for key in template_param_dict.keys()])
def param_dict(request):
    """all of the possible param_dicts generated by this fixture are invalid"""
    incorrect_dict = template_param_dict.copy()
    incorrect_dict[request.param] = None
    return incorrect_dict


def test_submit_rho_job_fails_on_bad_input(param_dict):
    """x"""
    # rjd.submit_rho_job(FS, param_dict) # need to create FS fi
    return