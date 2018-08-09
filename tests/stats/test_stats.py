""" """

# system imports
# import inspect
from functools import partial
import random
import string
import json
import os
# from os.path import dirname, join


# local imports
from ..context import pibronic
import pibronic.stats.stats as st
import pibronic.data.file_structure as fs
from pibronic.vibronic import vIO
from pibronic.constants import boltzman
from pibronic.pimc import BoxResultPM

# third party imports
import numpy as np
import pytest
from pytest import raises


def random_string(length):
    allchar = string.ascii_letters + string.punctuation + string.digits
    ret = "".join([random.choice(allchar) for x in range(length)])
    return ret


@pytest.fixture()
def root(tmpdir_factory):
    return tmpdir_factory.mktemp("root")


@pytest.fixture()
def temp_FS(root):
    MAX = 100
    id_data = random.randint(0, MAX)
    id_rho = random.randint(0, MAX)
    return fs.FileStructure(root, id_data=id_data, id_rho=id_rho)


def test_calculate_basic_property_terms():
    """obviously not a good test at the moment"""
    X = 100
    delta_beta = 1.0
    rho = np.ones(X)
    g = np.ones(X)
    g_plus = np.ones(X) + 1
    g_minus = np.ones(X)

    ret = st.calculate_basic_property_terms(delta_beta, rho, g, g_plus, g_minus)

    assert np.all(ret[0] == g / rho)
    assert np.all(ret[1] == 0.5)
    assert np.all(ret[2] == 1.0)
    return


def test_calculate_alpha_terms():
    """obviously not a good test at the moment"""
    X = 100
    delta_beta = 1.0
    rho = np.ones(X)
    g = np.ones(X)
    g_plus = np.ones(X) + 1
    g_minus = np.ones(X)
    alpha_plus = 2.0
    alpha_minus = 2.0

    ret = st.calculate_alpha_terms(delta_beta, rho, g, g_plus, g_minus, alpha_plus, alpha_minus)

    assert np.all(ret[0] == 1.0)
    assert ret[1] == 1.0
    assert ret[2] == 4.0
    assert np.all(ret[3] == 2.0)
    assert ret[4] == alpha_plus
    assert np.all(ret[5] == 1.0)
    assert ret[6] == alpha_minus
    return


def test_estimate_basic_properties():
    """obviously not a good test at the moment"""
    X = 100
    T = 300.00
    g_r = np.ones(X)
    sym1 = np.ones(X)
    sym2 = np.ones(X) + 1

    ret = st.estimate_basic_properties(X, T, g_r, sym1, sym2)
    kBT = boltzman * pow(T, 2.)

    assert ret["Z"] == 1.0
    assert ret["Z error"] == 0.0
    assert ret["E"] == -1.0
    assert ret["E error"] == 0.0
    assert ret["Cv"] == 1.0 / kBT
    assert ret["Cv error"] == 0.0
    return


def test_add_harmonic_contribution():
    nums = np.random.randint(0, 1000, size=4)
    test_dict = {"E": nums[0], "Cv": nums[1]}

    E_harmonic = nums[2]
    Cv_harmonic = nums[3]

    st.add_harmonic_contribution(test_dict, E_harmonic, Cv_harmonic)
    assert test_dict["E"] == nums[0] + nums[2]
    assert test_dict["Cv"] == nums[1] + nums[3]
    return


def test_apply_parameter_restrictions():
    """ this is just a placeholder function for a possible idea, it does nothing"""
    st.apply_parameter_restrictions(None)


def test_starmap_wrapper(monkeypatch, temp_FS):
    P = random.randint(0, 1000)
    T = random.uniform(5., 1000.)
    X = random.randint(0, int(1E6))

    def load_pimc_one(FS, P, T, pimc_results):
        assert isinstance(pimc_results, BoxResultPM)

    def load_analytical(FS, T, rhoData):
        assert not bool(rhoData) and isinstance(rhoData, dict)

    # monkeypatch the data loading functions
    monkeypatch.setattr('pibronic.data.postprocessing.load_pimc_data', load_pimc_one)
    monkeypatch.setattr('pibronic.data.postprocessing.load_analytic_data', load_analytical)

    # this is the 'fake' statistical operation we give to the starmap wrapper
    stat_op = lambda T, pimc_results, rhoData: {"test": True}

    # function should fail if FS doesn't have hash attributes
    with raises(AttributeError) as e_info:
        st.starmap_wrapper(temp_FS, P, T, stat_op)
    assert str(e_info.value) == f"FS {temp_FS:} doesn't have hash_vib or hash_rho attributes!"

    # so we give the FS hash attributes
    fake_file_length = 200
    temp_FS.hash_vib = vIO._hash(random_string(fake_file_length))
    temp_FS.hash_rho = vIO._hash(random_string(fake_file_length))

    # function should also fail if BoxResultPM has the default value of 0 samples
    with raises(AssertionError) as e_info:
        st.starmap_wrapper(temp_FS, P, T, stat_op)
    assert str(e_info.value) == "pimc_results has 0 samples! reading the data failed?!"

    # so set the BoxResultPM's samples attribute by monkeypatching the data loading function
    def load_pimc_two(FS, P, T, pimc_results):
        assert isinstance(pimc_results, BoxResultPM)
        pimc_results.samples = X  # give the BoxResultPM object a samples attribute

    monkeypatch.setattr('pibronic.data.postprocessing.load_pimc_data', load_pimc_two)

    # and now everything should work as expected
    st.starmap_wrapper(temp_FS, P, T, stat_op)
    path = temp_FS.template_jackknife.format(P=P, T=T, X=X)
    assert os.path.isfile(path)  # the file exists

    with open(path, mode='r', encoding='UTF8') as file:
        data = json.loads(file.read())

    # checking that the file has the appropriate minimum data
    assert bool(data["test"]) is True
    assert data["hash_vib"] == temp_FS.hash_vib
    assert data["hash_rho"] == temp_FS.hash_rho


class TestStatisticalAnalysisTypes():

    @pytest.fixture()
    def pimc_result(self):
        X = random.randint(0, int(1E3))
        result_obj = BoxResultPM(X=X)
        # fill the object with some random data
        result_obj.scaled_rho = np.array(random.sample(range(X), k=X))
        result_obj.scaled_g = np.array(random.sample(range(X), k=X))
        result_obj.scaled_gofr_plus = np.array(random.sample(range(X), k=X))
        result_obj.scaled_gofr_minus = np.array(random.sample(range(X), k=X))
        return result_obj

    def test_basic_statistical_analysis(self, monkeypatch, pimc_result):
        T = random.uniform(5., 1000.)
        X = pimc_result.samples

        # minimal testing data
        analytic_data = {"E": 1.0, "Cv": 2.0}
        test_terms = random.sample(range(1, 100), k=5)

        def monkey_one(db, *kwargs):
            assert np.allclose(kwargs[0], pimc_result.scaled_rho)
            assert np.allclose(kwargs[1], pimc_result.scaled_g)
            assert np.allclose(kwargs[2], pimc_result.scaled_gofr_plus)
            assert np.allclose(kwargs[3], pimc_result.scaled_gofr_minus)
            return test_terms

        def monkey_two(X_arg, T_arg, *terms):
            assert X_arg == X
            assert T_arg == T
            for t in terms:
                assert t in test_terms
            return {"E": 0.0, "Cv": 0.0}

        # monkeypatch the main functions to test simple execution flow
        monkeypatch.setattr('pibronic.stats.stats.calculate_basic_property_terms', monkey_one)
        monkeypatch.setattr('pibronic.stats.stats.estimate_basic_properties', monkey_two)

        # test
        basic_dict = st.basic_statistical_analysis(T, pimc_result, analytic_data)
        assert basic_dict["E"] == analytic_data["E"]
        assert basic_dict["Cv"] == analytic_data["Cv"]

    def test_alpha_statistical_analysis(self, monkeypatch, pimc_result):
        T = random.uniform(5., 1000.)
        X = pimc_result.samples

        # minimal testing data
        analytic_data = {"E": 1.0, "Cv": 2.0, "alpha_plus": 3.0, "alpha_minus": 4.0}
        test_terms = random.sample(range(1, 100), k=9)

        def monkey_one(db, *kwargs):
            assert np.allclose(kwargs[0], pimc_result.scaled_rho)
            assert np.allclose(kwargs[1], pimc_result.scaled_g)
            assert np.allclose(kwargs[2], pimc_result.scaled_gofr_plus)
            assert np.allclose(kwargs[3], pimc_result.scaled_gofr_minus)
            assert np.allclose(kwargs[4], analytic_data["alpha_plus"])
            assert np.allclose(kwargs[5], analytic_data["alpha_minus"])
            return test_terms

        def monkey_two(X_arg, T_arg, *terms):
            assert X_arg == X
            assert T_arg == T
            for t in terms:
                assert t in test_terms[:-4]  # don't get the last four things
            return {"E": 0.0, "Cv": 0.0}

        # monkeypatch the main functions to test simple execution flow
        monkeypatch.setattr('pibronic.stats.stats.calculate_alpha_terms', monkey_one)
        monkeypatch.setattr('pibronic.stats.stats.estimate_basic_properties', monkey_two)

        # test
        basic_dict = st.alpha_statistical_analysis(T, pimc_result, analytic_data)
        assert basic_dict["E"] == analytic_data["E"]
        assert basic_dict["Cv"] == analytic_data["Cv"]


def test_statistical_analysis_of_pimc(monkeypatch, temp_FS):

    test_list_pimc = random.sample(range(1, 100), k=10)
    test_list_beads = random.sample(range(1, 100), k=10)
    test_list_temps = random.sample(range(1, 100), k=10)

    # give the FS hash attributes
    fake_file_length = 200
    temp_FS.hash_vib = vIO._hash(random_string(fake_file_length))
    temp_FS.hash_rho = vIO._hash(random_string(fake_file_length))

    def monkey_one(FS):
        return test_list_pimc

    def monkey_two(list_pimc):
        assert list_pimc == test_list_pimc
        return test_list_beads

    def monkey_three(list_pimc):
        assert list_pimc == test_list_pimc
        return test_list_temps

    # monkeypatch the main functions to test simple execution flow
    monkeypatch.setattr('pibronic.data.postprocessing.retrive_pimc_file_list', monkey_one)
    monkeypatch.setattr('pibronic.data.postprocessing.extract_bead_paramater_list', monkey_two)
    monkeypatch.setattr('pibronic.data.postprocessing.extract_temperature_paramater_list', monkey_three)

    # had trouble pickling monkeypatched functions for starmap_wrapper, will look at later
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # def monkey_basic(FS, P, T, statistical_operation):
    #     assert FS is temp_FS
    #     assert P in test_list_beads
    #     assert T in test_list_temps
    #     assert statistical_operation == "basic"
    #     return

    # # first we test basic operation
    # method = "basic"
    # monkeypatch.setattr('pibronic.stats.stats.starmap_wrapper', monkey_basic)
    # st.statistical_analysis_of_pimc(temp_FS)
    # st.statistical_analysis_of_pimc(temp_FS, location="local")
    # st.statistical_analysis_of_pimc(temp_FS, method=method)
    # def monkey_alpha(FS, P, T, statistical_operation):
    #     assert FS is temp_FS
    #     assert P in test_list_beads
    #     assert T in test_list_temps
    #     assert statistical_operation == "alpha"
    #     return

    # # then we test alpha operation
    # method = "alpha"
    # monkeypatch.setattr('pibronic.stats.stats.starmap_wrapper', monkey_alpha)
    # st.statistical_analysis_of_pimc(temp_FS, method=method)

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------

    # function should fail because other operations don't exist
    method = 9
    with raises(Exception) as e_info:
        st.statistical_analysis_of_pimc(temp_FS, method=method)
    assert str(e_info.value) == f"Invalid value for parameter method:({method})"

    # function should fail because the server implementation doesn't exist yet
    with raises(AssertionError) as e_info:
        st.statistical_analysis_of_pimc(temp_FS, location="server")
    assert str(e_info.value) == "Need to write this code"

    # function should fail for an invalid location parameter
    location = 12
    with raises(Exception) as e_info:
        st.statistical_analysis_of_pimc(temp_FS, location=location)
    assert str(e_info.value) == f"Invalid value for parameter location:({location})"


def test_():
    pass


def test_():
    pass


def test_():
    pass


# TODO - should these be removed?
# @pytest.fixture()
# def path():
#     # does this work when deployed?
#     return join(dirname(dirname(inspect.getfile(pibronic))), "tests/test_models/")


# @pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
# def FS(path, request):
#     return fs.FileStructure(path, request.param[0], id_rho=request.param[1])
