""" """

# system imports
from os.path import join

# local imports
from .. import context
# from pibronic.vibronic import vIO
from pibronic.vibronic import electronic_structure as es

# third party imports
import pytest
from pytest import raises


@pytest.fixture()
def root(tmpdir_factory):
    return tmpdir_factory.mktemp("root")


@pytest.fixture()
def temp_file_path(root):
    p = root.join("tempfile.json")
    return str(p)


def test_State_enum():
    assert es.State.max() == es.State.Finished.value
    assert es.State.min() == es.State.OPT.value


class TestMemMappedHelperFunctions():
    """ lower priority for now, finish later"""

    def test_find_string_in_file():
        pass

    def test_rfind_string_in_file():
        pass

    def test_skip_back_n_lines():
        pass

    def test_skip_forward_n_lines():
        pass


def test_pretty_print_job_status():
    pass


def test_verify_file_exists(root, temp_file_path):
    fake_path = join(root, "fakefile")
    with raises(FileNotFoundError) as e_info:
        es.verify_file_exists(fake_path)
    assert str(e_info.value) == f"Cannot find {fake_path:}"

    assert es.verify_file_exists(temp_file_path)


def test_pairwise():
    pass


def test_get_yes_no_from_user():
    pass


def test_get_integer_from_user():
    pass


def test_wait_on_results():
    pass


def test_verify_aces2_completed():
    pass


def test_submit_job():
    pass


def test_parse_input_parameters():
    pass


def test_hartree_fock_calculation():
    pass


def test__estimate_dropmo():
    pass


def test__extract_lines_for_estimating_NIP():
    pass


def test__guess_NIP_value():
    pass
