""" """

# system imports
from os.path import join
import itertools as it
import logging
import random
import mmap

# local imports
from .. import context
# from pibronic.log_conf import log
# from pibronic.vibronic import vIO
from pibronic.data import file_structure as fs
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
    p.write("")  # create an empty file
    return str(p)  # return the path to that file


def test_State_enum():
    assert es.State.max() == es.State.FINISHED.value
    assert es.State.min() == es.State.OPT.value


class TestMemMappedHelperFunctions():
    """ lower priority for now, finish later"""

    # the file will contain 10 lines of this string
    test_string = "this is a test, this is a test two\n" * 10
    test_bytes = test_string.encode()

    @pytest.fixture()
    def path(self, root):
        p = root.join("tempfile.json")
        p.write(self.test_string)  # write the data to the file
        return str(p)  # return the path to that file

    @pytest.fixture()
    def memmap_file(self, path):
        with open(path, "r") as file:
            with mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:
                yield memmap_file

    def test_find_string_in_file(self, path, memmap_file):
        # test finding a single substring
        target_string = "two"
        loc = es.find_string_in_file(memmap_file, path, "two")
        assert loc == self.test_bytes.find(target_string.encode())

        # test finding the first of two substrings
        target_string = "this"
        loc = es.find_string_in_file(memmap_file, path, "this")
        assert loc == self.test_bytes.find(target_string.encode())

        # test not finding a substring
        target_string = "lobster"
        with raises(Exception) as e_info:
            es.find_string_in_file(memmap_file, path, target_string)
        assert f"It seems \"{target_string:s}\" was not present in the file" in str(e_info.value)

    def test_rfind_string_in_file(self, path, memmap_file):
        # test finding a single substring
        target_string = "two"
        loc = es.rfind_string_in_file(memmap_file, path, "two")
        assert loc == self.test_bytes.rfind(target_string.encode())

        # test finding the second of two substrings
        target_string = "this"
        loc = es.rfind_string_in_file(memmap_file, path, "this")
        assert loc == self.test_bytes.rfind(target_string.encode())

        # test not finding a substring
        target_string = "lobster"
        with raises(Exception) as e_info:
            es.rfind_string_in_file(memmap_file, path, target_string)
        assert f"It seems \"{target_string:s}\" was not present in the file" in str(e_info.value)

    def test_skip_back_n_lines(self, memmap_file):
        # test going back 0 lines
        n_lines = 0
        start_index = len(self.test_bytes) // 2  # start somewhere in the middle
        loc = es.skip_back_n_lines(memmap_file, n_lines, start_index)
        assert loc == start_index

        # test going back 2 lines
        n_lines = 2
        start_index = len(self.test_bytes) // 2  # start somewhere in the middle
        loc = es.skip_back_n_lines(memmap_file, n_lines, start_index)
        for _ in it.repeat(None, n_lines):
            index = self.test_bytes.rfind(b'\n', 0, start_index)
            start_index = index
        assert loc == index

        # test going back to the start of the file
        n_lines = 10
        start_index = len(self.test_bytes)  # start at the end of the file
        loc = es.skip_back_n_lines(memmap_file, n_lines, start_index)
        for _ in it.repeat(None, n_lines):
            index = self.test_bytes.rfind(b'\n', 0, start_index)
            start_index = index
        assert loc == index

    def test_skip_forward_n_lines(self, memmap_file):
        # test going forward 0 lines
        n_lines = 0
        start_index = len(self.test_bytes) // 2  # start somewhere in the middle
        loc = es.skip_forward_n_lines(memmap_file, n_lines, start_index)
        assert loc == start_index

        # test going forward 2 lines
        n_lines = 2
        start_index = len(self.test_bytes) // 2  # start somewhere in the middle
        loc = es.skip_forward_n_lines(memmap_file, n_lines, start_index)
        for _ in it.repeat(None, n_lines):
            index = self.test_bytes.find(b'\n', start_index + 1)
            start_index = index
        assert loc == index

        # test going forward to the end of the file
        n_lines = 10
        start_index = 0  # start at the beginning of the file
        loc = es.skip_forward_n_lines(memmap_file, n_lines, start_index)
        for _ in it.repeat(None, n_lines):
            index = self.test_bytes.find(b'\n', start_index + 1)
            start_index = index
        assert loc == index


def test_pretty_print_job_status(root, caplog):
    # caplog.set_level(logging.INFO, logger=log)  # capture the logs

    # make a FileStructure object for one of the models
    model_idx = random.randint(0, len(es.number_dictionary.keys()) - 1)
    model = es.number_dictionary[model_idx]

    # write a template file
    FS = fs.FileStructure(root, id_data=model[1], id_rho=0)
    path = join(FS.path_es, es.name_of_state_file)
    with open(path, 'w') as file:
        file.write("OPT\nNIP\nVIB\nIPEOMCC\nPREPVIB\n")

    # check that output works correctly
    es.pretty_print_job_status(FS.path_root)
    for record in caplog.records:
        message = record.getMessage()
        if str(model[0]) in message and str(model[1]) in message:
            assert f"({model[0]:}, {model[1]:}) last state was PREPVIB" in message


def test_verify_file_exists(root, temp_file_path):
    fake_path = join(root, "fakefile")
    with raises(FileNotFoundError) as e_info:
        es.verify_file_exists(fake_path)
    assert str(e_info.value) == f"Cannot find {fake_path:}"
    assert es.verify_file_exists(temp_file_path)


def test_pairwise():
    iterable = range(random.randint(10, 30))
    zipped = es.pairwise(iterable)
    for a, b in zipped:
        pass


def test_get_yes_no_from_user(monkeypatch):
    # monkeypatch the "input" function, so that it returns "yes".
    # This simulates the user entering "yes" in the terminal:
    monkeypatch.setattr('builtins.input', lambda x: "yes")
    assert es.get_yes_no_from_user("")

    # monkeypatch the "input" function, so that it doesn't return "yes".
    monkeypatch.setattr('builtins.input', lambda x: "blah")
    assert not es.get_yes_no_from_user("")


def test_get_integer_from_user(monkeypatch):
    # monkeypatch the "input" function, so that it returns 12.
    monkeypatch.setattr('builtins.input', lambda x: 12)
    assert es.get_integer_from_user("")


def test_wait_on_results(monkeypatch):
    job_id = random.randint(0, 100)  # some number
    job_type = "blue"  # some string

    def check_sync(arg_job_id, arg_job_type):
        assert arg_job_id is job_id
        assert arg_job_type is job_type

    def result_FAILED(arg_job_id):
        assert arg_job_id is job_id
        return "FAILED"

    def result_RUNNING(arg_job_id):
        assert arg_job_id is job_id
        return "RUNNING"

    def result_COMPLETED(arg_job_id):
        assert arg_job_id is job_id
        return "COMPLETED"

    monkeypatch.setattr('pibronic.server.job_boss.synchronize_with_job', check_sync)
    monkeypatch.setattr('time.sleep', lambda n: None)

    # check failed
    monkeypatch.setattr('pibronic.server.job_boss.check_acct_state', result_FAILED)
    with raises(Exception) as e_info:
        es.wait_on_results(job_id, job_type)
    assert str(e_info.value) == f"{job_type:s} script (JOBID={job_id:d}) failed for some reason"

    # check RUNNING
    monkeypatch.setattr('pibronic.server.job_boss.check_acct_state', result_RUNNING)
    with raises(Exception) as e_info:
        es.wait_on_results(job_id, job_type)
    assert str(e_info.value) == f"unknown result in {job_type:s} parsing"

    # check COMPLETED
    monkeypatch.setattr('pibronic.server.job_boss.check_acct_state', result_COMPLETED)
    es.wait_on_results(job_id, job_type)


class Test_verify_aces2_completed():

    @pytest.fixture()
    def temp_aces2_successful_path(self, root):
        p = root.join("tempfile.json")
        test_string = "wordswordswords\n" * 20  # file must be longer than 10 lines
        p.write(test_string)  # create an empty file
        return str(p)  # return the path to that file

    @pytest.fixture()
    def temp_aces2_successful_path(self, root):
        p = root.join("tempfile.json")
        test_string = "wordswordswords\n" * 20  # file must be longer than 10 lines
        p.write(test_string)  # create an empty file
        return str(p)  # return the path to that file

    def test_verify_aces2_completed_SUCCESS(self, monkeypatch, caplog, root):
        # found the successful job string
        job_id = random.randint(0, 100)  # some number
        p = root.join("tmpfile")
        string = "wordswordswords\n" * 20
        string += 'The ACES2 program has completed successfully in'
        p.write(string)
        path = str(p)

        monkeypatch.setattr('pibronic.server.job_boss.check_slurm_output', lambda x, y: None)
        es.verify_aces2_completed(root, path, job_id)
        for record in caplog.records:
            message = record.getMessage()
            assert len(message) is 0

    def test_verify_aces2_completed_SLURM_FAILED(self, monkeypatch, caplog, root):
        # found the successful job string
        job_id = random.randint(0, 100)  # some number
        p = root.join("tmpfile")
        string = "wordswordswords\n" * 20
        p.write(string)
        path = str(p)

        def AssertFalse(x, y):
            assert False

        monkeypatch.setattr('pibronic.server.job_boss.check_slurm_output', AssertFalse)

        with raises(AssertionError) as e_info:
            es.verify_aces2_completed(root, path, job_id)

        for record in caplog.records:
            message = record.getMessage()
            assert "did not complete successfully" in message

    def test_verify_aces2_completed_ACES2_FAILED(self, monkeypatch, caplog, root):
        job_id = random.randint(0, 100)  # some number
        p = root.join("tmpfile")
        string = "wordswordswords\n" * 20
        p.write(string)
        path = str(p)

        monkeypatch.setattr('pibronic.server.job_boss.check_slurm_output', lambda x, y: None)

        with raises(Exception) as e_info:
            es.verify_aces2_completed(root, path, job_id)

        assert "Could not find an issue with slurm output" in str(e_info.value)


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


def test__estimate_NIP():
    pass


def test_parse_hartree_fock_output():
    pass


def test_geometry_optimization():
    pass


def test_parse_opt_output():
    pass


def test_extract_internal():
    pass


def test_extract_cartesian():
    pass


def test_vibrational_frequency():
    pass


def test_ip_calculation():
    pass


def test_parse_ip_output():
    pass


def test_verify_percent_singles():
    pass


def test_verify_ip_states():
    pass


def test__create_template():
    pass


def test_prepVibron_calculation():
    pass


def test__remove_extra_heff():
    pass


def test__copy_output_file():
    pass


def test__add_nip_to_file():
    pass


def test_parse_prepVibron_output():
    pass


def test_vibron_calculation():
    pass


def test_parse_vibron_output():
    pass


class TestZmatClass():

    def test_(self):
        pass


class TestVibronExecutionClass():

    def test_(self):
        pass


def test_test_one():
    pass


def test_calculate_vibronic_model_wrapper_one():
    pass


def test_main():
    pass
