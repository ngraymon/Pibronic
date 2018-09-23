"""vibronic_model_io.py should handle the majority of file I/O"""

# system imports
import itertools as it
import hashlib
import random
import shutil
import copy
import json
import os
from os.path import join, isfile

# third party imports
import numpy as np
from numpy import float64 as F64
from numpy.random import uniform as Uniform

# local imports
from ..log_conf import log
# from .. import constants
# from ..data import file_structure
# from ..data import file_name
from .vibronic_model_keys import VibronicModelKeys as VMK
from . import model_auto
from . import model_op
from . import model_h
from . import orthonormal

np.set_printoptions(precision=8, suppress=True)  # Print Precision!

# TODO - made the design decision that if a key is not present in the json file that implies all the values are zero
# - need to make sure this is enforced across all the code


def model_shape_dict(A, N):
    """ returns a dictionary with the same keys as the .json file whose values are tuples representing the dimensionality of the associated value in the .json file
    Takes A - number of surfaces and N - number of modes

    """
    dictionary = {
                  VMK.E:  (A, A),
                  VMK.w:  (N, ),
                  VMK.G1: (N, A, A),
                  VMK.G2: (N, N, A, A),
                  VMK.G3: (N, N, N, A, A),
                  VMK.G4: (N, N, N, N, A, A),
                  }

    return dictionary


def diagonal_model_shape_dict(A, N):
    """ returns a dictionary with the same keys as the .json file whose values are tuples representing the dimensionality of the associated value in the .json file
    Takes A - number of surfaces and N - number of modes

    """
    dictionary = {
                  VMK.E:  (A, ),
                  VMK.w:  (N, ),
                  VMK.G1: (N, A),
                  VMK.G2: (N, N, A),
                  VMK.G3: (N, N, N, A),
                  VMK.G4: (N, N, N, N, A),
                  }

    return dictionary


def _array_is_symmetric_in_A(array):
    """ Boolean function that returns true if the provided numpy array is symmetric in the surface dimension
    where the surface dimensions (A) are by convention the last two dimensions
    this function assumes that the array is properly formatted
    """

    new_dims = list(range(array.ndim))
    # swap the last two dimensions, which by convention are the surface dimensions
    new_dims[-1], new_dims[-2] = new_dims[-2], new_dims[-1]

    return np.allclose(array, array.transpose(new_dims))


def model_parameters_are_symmetric_in_surfaces(kwargs):
    """ Boolean function that returns true if the provided model's arrays are all symmetric in the surface dimension
    where the surface dimensions (A) are by convention the last two dimensions
    this function assumes that the arrays is properly formatted
    """
    verify_model_parameters(kwargs)

    for key in kwargs.keys():
        if isinstance(kwargs[key], np.ndarray) and kwargs[key].ndim >= 2:
            if not _array_is_symmetric_in_A(kwargs[key]):
                log.debug(f"{key} not symmetric in surfaces")
                return False

    return True


def model_parameters_are_symmetric_in_modes(kwargs):
    """ Boolean function that returns true if the provided model's quadratic and quartic arrays are symmetric in their mode dimensions
    this is a bit trickier than the surface dimension
    this function assumes that the arrays is properly formatted
    """
    verify_model_parameters(kwargs)

    # do the quadratic case
    key = VMK.G2
    if key in kwargs.keys():
        new_dims = list(range(kwargs[key].ndim))
        new_dims[0], new_dims[1] = new_dims[1], new_dims[0]
        if not np.allclose(kwargs[key], kwargs[key].transpose(new_dims)):
            return False

    # I don't think there is a general approach for the cubic case

    # do the quartic case
    key = VMK.G4
    if key in kwargs.keys():
        new_dims = list(range(kwargs[key].ndim))
        new_dims[0], new_dims[1] = new_dims[1], new_dims[0]
        new_dims[2], new_dims[3] = new_dims[3], new_dims[2]
        if not np.allclose(kwargs[key], kwargs[key].transpose(new_dims)):
            return False

    return True


def model_zeros_template_json_dict(A, N):
    """ returns a dictionary that is a valid model, where all values (other than states and modes) are set to 0
    """
    shape = model_shape_dict(A, N)
    dictionary = {
                  VMK.N: N,
                  VMK.A: A,
                  VMK.E: np.zeros(shape[VMK.E], dtype=F64),
                  VMK.w: np.zeros(shape[VMK.w], dtype=F64),
                  VMK.G1: np.zeros(shape[VMK.G1], dtype=F64),
                  VMK.G2: np.zeros(shape[VMK.G2], dtype=F64),
                  VMK.G3: np.zeros(shape[VMK.G3], dtype=F64),
                  VMK.G4: np.zeros(shape[VMK.G4], dtype=F64),
                  }

    return dictionary


def diagonal_model_zeros_template_json_dict(A, N):
    """ returns a dictionary that is a valid diagonal model, where all values (other than states and modes) are set to 0
    """
    shape = diagonal_model_shape_dict(A, N)
    dictionary = {
                  VMK.N: N,
                  VMK.A: A,
                  VMK.E: np.zeros(shape[VMK.E], dtype=F64),
                  VMK.w: np.zeros(shape[VMK.w], dtype=F64),
                  VMK.G1: np.zeros(shape[VMK.G1], dtype=F64),
                  VMK.G2: np.zeros(shape[VMK.G2], dtype=F64),
                  VMK.G3: np.zeros(shape[VMK.G3], dtype=F64),
                  VMK.G4: np.zeros(shape[VMK.G4], dtype=F64),
                  }

    return dictionary


def verify_model_parameters(kwargs):
    """make sure the provided model parameters follow the file conventions"""
    assert VMK.N in kwargs, "need the number of modes"
    assert VMK.A in kwargs, "need the number of surfaces"

    A, N = _extract_dimensions_from_dictionary(kwargs)
    shape_dict = model_shape_dict(A, N)

    for key, value in kwargs.items():
        if key in shape_dict:
            assert kwargs[key].shape == shape_dict[key], f"{key} have incorrect shape"
        else:
            log.debug(f"Found key {key} which is not present in the default dictionary")

    return


def verify_diagonal_model_parameters(kwargs):
    """make sure the provided sample parameters follow the file conventions"""
    assert VMK.N in kwargs, "need the number of modes"
    assert VMK.A in kwargs, "need the number of surfaces"

    A, N = _extract_dimensions_from_dictionary(kwargs)
    shape_dict = diagonal_model_shape_dict(A, N)

    for key, value in kwargs.items():
        if key in shape_dict:
            assert kwargs[key].shape == shape_dict[key], f"{key} have incorrect shape"
        else:
            log.debug(f"Found key {key} which is not present in the default dictionary")

    return


def _same_model(d1, d2):
    """ returns True if all parameters of the two dictionaries have the same dimensions
    and the same floating point numbers up to standard precision comparison
    raises an assertion error if either dictionary has incorrect parameters"""

    verify_model_parameters(d1)
    verify_model_parameters(d2)

    A1, N1 = _extract_dimensions_from_dictionary(d1)
    A2, N2 = _extract_dimensions_from_dictionary(d1)

    if A1 != A2 or N1 != N2:
        return False

    new_d1 = model_zeros_template_json_dict(A1, N1)
    new_d1.update(d1)

    new_d2 = model_zeros_template_json_dict(A2, N2)
    new_d2.update(d2)

    for key in new_d1.keys():
        if not np.allclose(new_d1[key], new_d2[key]):
            log.debug(f"These models differ for key {key}\nd1: {new_d1[key]}\nd2: {new_d2[key]}")
            return False
        # elif (key not in d2 and np.count_nonzero(d1[key]) > 0) or \
        #      (key not in d1 and np.count_nonzero(d2[key]) > 0):
        #     print(f"These models differ for key {key}\nd1: {d1[key]}\nd2: {d2[key]}")
        #     return False

    else:
        return True

    raise Exception("This line of code should not be reached!?")


def _same_diagonal_model(d1, d2):
    """ returns True if all parameters of the two dictionaries have the same dimensions
    and the same floating point numbers up to standard precision comparison
    raises an assertion error if either dictionary has incorrect parameters"""

    verify_diagonal_model_parameters(d1)
    verify_diagonal_model_parameters(d2)

    A1, N1 = _extract_dimensions_from_dictionary(d1)
    A2, N2 = _extract_dimensions_from_dictionary(d1)

    if A1 != A2 or N1 != N2:
        return False

    new_d1 = diagonal_model_zeros_template_json_dict(A1, N1)
    new_d1.update(d1)

    new_d2 = diagonal_model_zeros_template_json_dict(A2, N2)
    new_d2.update(d2)

    for key in new_d1.keys():
        if not np.allclose(new_d1[key], new_d2[key]):
            print(f"These diagonal models differ for key {key}\nd1: {new_d1[key]}\nd2: {new_d2[key]}")
            return False
    else:
        return True

    raise Exception("This line of code should not be reached!?")


def create_random_model():
    """ returns a dictionary that is a valid model
    """
    d = {'numStates': random.randint(2, 10),
         'numModes': random.randint(2, 20),
         'quadratic_scaling': random.uniform(0.04, 0.12),
         'linear_scaling': random.uniform(0.02, 0.06),
         'diagonal': False,
         }
    return generate_vibronic_model_data(d)


def create_random_diagonal_model():
    """ returns a dictionary that is a valid diagonal model
    """
    d = {'numStates': random.randint(2, 10),
         'numModes': random.randint(2, 20),
         'quadratic_scaling': random.uniform(0.04, 0.12),
         'linear_scaling': random.uniform(0.02, 0.06),
         'diagonal': True,
         }
    return generate_vibronic_model_data(d)


def _hash(string):
    """creates a SHA512 hash of the input string and returns the byte representation"""
    m = hashlib.sha512()
    m.update(string.encode('UTF-8'))
    return m.hexdigest()


def create_model_hash(FS=None, path=None):
    """ create a hash of the coupled_model.json file's contents
    this is used to confirm that result files were generated for the current model and not an older one
    uses a FileStructure or an absolute path to the file"""
    if FS is None:
        assert path is not None, "no arguments provided"
    else:
        path = FS.path_vib_model

    assert isfile(path)

    with open(path, mode='r', encoding='UTF8') as file:
        string = file.read()

    return _hash(string)


def create_diagonal_model_hash(FS=None, path=None):
    """ create a hash of the sampling_model.json file's contents
    this is used to confirm that result files were generated for the current model and not an older one
    uses a FileStructure or an absolute path to the file"""
    if FS is None:
        assert path is not None, "no arguments provided"
    else:
        path = FS.path_rho_model

    assert isfile(path)

    with open(path, mode='r', encoding='UTF8') as file:
        string = file.read()

    return _hash(string)


def _generate_linear_terms(linear_terms, shape, displacement, Modes):
    """ generate linear terms that are 'reasonable' """
    for i in Modes:
        upTri = Uniform(-displacement[i], displacement[i], shape[VMK.E])
        # force the linear terms to be symmetric
        linear_terms[i, ...] = np.tril(upTri) + np.tril(upTri, k=-1).T
    return


def _generate_quadratic_terms(quadratic_terms, shape, displacement, Modes):
    """ generate quadratic terms that are 'reasonable' """
    for i, j in it.product(Modes, repeat=2):
        upTri = Uniform(-displacement[i, j], displacement[i, j], shape[VMK.E])
        # force the quadratic terms to be symmetric
        quadratic_terms[i, j, ...] = np.tril(upTri) + np.tril(upTri, k=-1).T
        quadratic_terms[j, i, ...] = np.tril(upTri) + np.tril(upTri, k=-1).T
    return


def generate_vibronic_model_data(input_parameters=None):
    """redo this one but otherwise its fine returns e,w,l,q filled with appropriate values"""
    paramDict = {   # default values
                    'frequency_range': [0.02, 0.04],
                    'energy_range': [0.0, 2.0],
                    'quadratic_scaling': 0.08,
                    'linear_scaling': 0.04,
                    'diagonal': False,
                    'numStates': 2,
                    'numModes': 3,
                    }

    if input_parameters is not None:
        paramDict.update(input_parameters)

    # readability
    minE, maxE = paramDict['energy_range']
    minFreq, maxFreq = paramDict['frequency_range']

    # ranges for convenience
    numModes = paramDict['numModes']
    numStates = paramDict['numStates']
    Modes = range(numModes)

    # generate the array dimensions
    shape = model_shape_dict(numStates, numModes)

    # assume we are building a coupled model
    model = model_zeros_template_json_dict(numStates, numModes)

    # generate frequencies
    model[VMK.w] = np.linspace(minFreq, maxFreq, num=numModes, endpoint=True, dtype=F64)

    # generate energy
    model[VMK.E] = Uniform(minE, maxE, shape[VMK.E])
    # force the energy to be symmetric
    model[VMK.E] = np.tril(model[VMK.E]) + np.tril(model[VMK.E], k=-1).T

    # calculate the linear displacement
    l_shift = paramDict['linear_scaling'] / model[VMK.w]
    _generate_linear_terms(model[VMK.G1], shape, l_shift, Modes)

    # TODO - no quadratic terms for the moment - turn back on after further testing
    # calculate the quadratic displacement
    # q_shift = np.sqrt(np.outer(frequencies, frequencies)) / paramDict['quadratic_scaling']
    # _generate_quadratic_terms(q_terms, shape, q_shift, Modes)

    # if we are building a harmonic model then zero out all off-diagonal entries
    if paramDict['diagonal']:
        d_model = diagonal_model_zeros_template_json_dict(numStates, numModes)

        d_model[VMK.E] = np.diag(model[VMK.E])
        d_model[VMK.w] = model[VMK.w]
        for i in Modes:
            d_model[VMK.G1][i, ...] = np.diag(model[VMK.G1][i, ...])
        for i, j in it.product(Modes, repeat=2):
            d_model[VMK.G2][i, j, ...] = np.diag(model[VMK.G2][i, j, ...])

        return d_model

    assert model_parameters_are_symmetric_in_surfaces(model)
    assert model_parameters_are_symmetric_in_modes(model)

    return model


def read_model_h_file(path_file_h):
    """ wrapper function to maintain functionality - possible remove in the future"""
    return model_h.read_model_h_file(path_file_h)


def read_model_op_file(path_file_op):
    """ wrapper function to maintain functionality - possible remove in the future"""
    return model_op.read_model_op_file(path_file_op)


def create_coupling_from_h_file(FS, path_file_h):
    """assumes that the path_file_h is in electronic_structure"""
    # A, N, E, w, l, q = read_model_h_file(path_file_h)
    model_dict = read_model_h_file(path_file_h)
    save_model_to_JSON(FS.path_vib_model, model_dict)
    log.debug("Created \n{:s}\nfrom\n{:s}\n".format(FS.path_vib_model, path_file_h))
    return FS.path_vib_model


def create_coupling_from_op_file(FS, path_file_op):
    """assumes that the path_file_op is in electronic_structure"""
    # A, N, E, w, l, q = read_model_h_file(path_file_op)
    model_dict = read_model_op_file(path_file_op)
    save_model_to_JSON(FS.path_vib_model, model_dict)
    log.debug("Created \n{:s}\nfrom\n{:s}\n".format(FS.path_vib_model, path_file_op))
    return FS.path_vib_model


def create_coupling_from_op_hyperlink(FS, url):
    """assumes that the path_file_op is in electronic_structure"""
    import urllib

    assert False, "This function is currently unfinished"

    request = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(request)
        # response is now a string you can search through containing the page's html
        # A, N, E, w, l, q = read_model_h_file(path_file_op)
        # args = read_model_op_file(path_file_op)  # need to make a choice about the function here
        args = []
        save_model_to_JSON(FS.path_vib_model, args)
        log.debug("Created \n{:s}\nfrom\n{:s}\n".format(FS.path_vib_model, url))
        return FS.path_vib_model
    except Exception as err:
        # The URL wasn't valid
        raise Exception("Incorrect https link {:s}".format(url))


def read_model_auto_file(path_file_auto):
    """ wrapper function to maintain functionality - possible remove in the future"""
    return model_auto.read_model_auto_file(path_file_auto)


def _extract_dimensions_from_dictionary(dictionary):
    """x"""
    N = int(dictionary[VMK.N])
    A = int(dictionary[VMK.A])
    return A, N


def _extract_dimensions_from_file(path):
    """x"""
    assert isfile(path), f"invalid path:\n{path:s}"
    with open(path, mode='r', encoding='UTF8') as file:
        input_dictionary = json.loads(file.read())
    VMK.change_dictionary_keys_from_strings_to_enum_members(input_dictionary)
    return _extract_dimensions_from_dictionary(input_dictionary)


def extract_dimensions_of_model(FS=None, path=None):
    """return number_of_modes and number_of_surfaces for coupling_model.json files by using a FileStructure or an absolute path to the file"""
    if FS is None:
        assert path is not None, "no arguments provided"
    else:
        path = FS.path_vib_model
    return _extract_dimensions_from_file(path)


def extract_dimensions_of_diagonal_model(FS=None, path=None):
    """return number_of_modes and number_of_surfaces for sampling_model.json files by using a FileStructure or an absolute path to the file
    """

    """ TODO - it might be nice to have the ability to specify id_data or id_rho, although this should be done in a way that queries file_structure so as to not "leak" the file structure out to other areas of the code
    """

    if FS is None:
        assert path is not None, "no arguments provided"
    else:
        path = FS.path_rho_model
    return _extract_dimensions_from_file(path)


def _save_to_JSON(path, dictionary):
    dict_copy = copy.deepcopy(dictionary)
    VMK.change_dictionary_keys_from_enum_members_to_strings(dict_copy)
    """ converts each numpy array to a list so that json can serialize them properly"""
    for key, value in list(dict_copy.items()):
        if isinstance(value, (np.ndarray, np.generic)):
            if np.count_nonzero(value) > 0:
                dict_copy[key] = value.tolist()
            else:
                del dict_copy[key]
        else:
            log.debug(f"Value {value} with Key {key} does not appear to be an ndarray")

    with open(path, mode='w', encoding='UTF8') as target_file:
        target_file.write(json.dumps(dict_copy))

    return


def save_model_to_JSON(path, dictionary):
    """ wrapper for _save_to_JSON
    calls verify_model_parameters() before calling _save_to_JSON()
    """
    verify_model_parameters(dictionary)
    log.debug(f"Saving model to {path:s}")
    _save_to_JSON(path, dictionary)
    return


def save_diagonal_model_to_JSON(path, dictionary):
    """ wrapper for _save_to_JSON
    calls verify_sample_parameters() before calling _save_to_JSON()
    """
    verify_diagonal_model_parameters(dictionary)
    log.debug(f"Saving sample to {path:s}")
    _save_to_JSON(path, dictionary)
    return


def _load_inplace_from_JSON(path, dictionary):
    """overwrites all provided values in place with the values stored in the .json file located at path"""

    with open(path, mode='r', encoding='UTF8') as file:
        input_dictionary = json.loads(file.read())

    VMK.change_dictionary_keys_from_strings_to_enum_members(input_dictionary)

    for key, value in dictionary.items():
        if isinstance(value, (np.ndarray, np.generic)):
            # this is a safer way of forcing the input arrays that have no corresponding key in the input_dictionary to have zero values
            # although this might not be necessary, it is a safer alternative at the moment
            if key not in input_dictionary:
                dictionary[key].fill(0.0)
            else:
                dictionary[key][:] = np.array(input_dictionary[key], dtype=F64)
    return


def _load_from_JSON(path):
    """returns a dictionary filled with the values stored in the .json file located at path"""

    with open(path, mode='r', encoding='UTF8') as file:
        input_dictionary = json.loads(file.read())

    VMK.change_dictionary_keys_from_strings_to_enum_members(input_dictionary)

    for key, value in input_dictionary.items():
        if isinstance(value, list):
            # if we don't predefine the shape, can we run into problems?
            input_dictionary[key] = np.array(value, dtype=F64)

    # special case to always create an array of energies that are 0.0 if not provided in the .json file
    if VMK.E not in input_dictionary:
        A, N = _extract_dimensions_from_dictionary(input_dictionary)
        shape = model_shape_dict(A, N)
        input_dictionary[VMK.E] = np.zeros(shape[VMK.E], dtype=F64)

    # TODO - design decision about which arrays to fill with zeros by default?

    return input_dictionary


def load_model_from_JSON(path, dictionary=None):
    """
    if kwargs is not provided then returns a dictionary filled with the values stored in the .json file located at path

    if kwargs is provided then all values are overwritten (in place) with the values stored in the .json file located at path
    """
    log.debug(f"Loading model from {path:s}")

    # no arrays were provided so return newly created arrays after filling them with the appropriate values
    if not bool(dictionary):
        new_model_dict = _load_from_JSON(path)

        # TODO - we might want to make sure that none of the values in the dictionary have all zero values or are None

        verify_model_parameters(new_model_dict)
        return new_model_dict

    # arrays were provided so fill them with the appropriate values
    else:
        verify_model_parameters(dictionary)
        _load_inplace_from_JSON(path, dictionary)
        # check twice? might as well be cautious for the moment until test cases are written
        verify_model_parameters(dictionary)

    return


def load_diagonal_model_from_JSON(path, dictionary=None):
    """
    if kwargs is not provided then returns a dictionary filled with the values stored in the .json file located at path

    if kwargs is provided then all values are overwritten (in place) with the values stored in the .json file located at path
    """
    log.debug(f"Loading rho model (sampling model) from {path:s}")

    # no arrays were provided so return newly created arrays after filling them with the appropriate values
    if not bool(dictionary):
        new_model_dict = _load_from_JSON(path)

        # TODO - we might want to make sure that none of the values in the dictionary have all zero values or are None

        verify_diagonal_model_parameters(new_model_dict)
        return new_model_dict

    # arrays were provided so fill them with the appropriate values
    else:
        verify_diagonal_model_parameters(dictionary)
        _load_inplace_from_JSON(path, dictionary)
        # check twice? might as well be cautious for the moment until test cases are written
        verify_diagonal_model_parameters(dictionary)
    return
# ------------------------------------------------------------------------


def simple_single_point_energy_calculation(FS, path):
    """ generate new energy values for the diagonal model stored at the arg 'path'
    this is intended to be used for re-weighting the oscillators created using the iterative method

    returns a list of energy values in eV which"""

    iterative_model = load_diagonal_model_from_JSON(path)
    assert VMK.G2 not in iterative_model, "doesn't support quadratic terms at the moment"
    Ai, Ni = _extract_dimensions_from_dictionary(iterative_model)

    # generate the oscillator minimum's
    minimums = np.zeros((Ai, Ni))

    # if there is no linear term parameter in the model's dictionary, then they are all zero, and therefore are centered at the origin.
    if VMK.G1 in iterative_model:
        w_iter = iterative_model[VMK.w]
        lin_iter = iterative_model[VMK.G1]
        for idx in range(Ai):
            minimums[idx, :] = np.divide(-lin_iter[:, idx],  w_iter[:])

    coupled_model = load_model_from_JSON(FS.path_vib_model)
    A, N = _extract_dimensions_from_dictionary(coupled_model)
    assert VMK.G2 not in coupled_model, "doesn't support quadratic terms at the moment"
    E = coupled_model[VMK.E]
    w = coupled_model[VMK.w]
    lin = np.zeros(model_shape_dict(A, N)[VMK.G1])
    if VMK.G1 in coupled_model:
        lin[:] = coupled_model[VMK.G1]
    # quad = coupled_model[VMK.G2]  # TODO - add support for quadratic terms in the future

    new_energy_values = np.zeros(Ai)

    for idx in range(Ai):
        q = minimums[idx, :]  # we evaluate the (R/q) point

        # compute the harmonic oscillator contribution
        ho = np.zeros((A, A))
        np.fill_diagonal(ho, np.sum(w * pow(q, 2)))
        ho *= 0.5

        V = np.zeros((A, A))
        V[:] += E
        V[:] += ho

        for a1 in range(A):
            for a2 in range(A):
                V[a1, a2] += np.sum(lin[:, a1, a2] * q)

        eigenvalues = np.linalg.eigvalsh(V)
        new_energy_values[idx] = min(eigenvalues)

    return new_energy_values


def recalculate_energy_values_of_diagonal_model(FS, path):
    """ x """

    """ note that these new energy values \tilde{E}^{a} values defined in equation 34 on page 4 from the archive paper, however the energy values stored in the *_model.json files are the E^{aa'} values in equation 26 on page 4 so we must subtract the \Delta^{a} term from the 'new_energies' to obtain the correct energy values which will be stored in the *_model.json file
    """
    new_energies = simple_single_point_energy_calculation(FS, path)

    # modify the model located at the provided path
    model = load_diagonal_model_from_JSON(path)

    # TODO - should we force that the linear terms are always present in any loaded model just like the energy values? then we don't need these checks.
    A, N = _extract_dimensions_from_dictionary(model)
    lin = np.zeros(diagonal_model_shape_dict(A, N)[VMK.G1])
    if VMK.G1 in model:
        lin[:] = model[VMK.G1]

    delta_a = -0.5 * (lin**2. / model[VMK.w][:, np.newaxis]).sum(axis=0)
    model[VMK.E] = new_energies - delta_a
    save_diagonal_model_to_JSON(path, model)
    return


def fill_offdiagonal_of_model_with_zeros(model):
    """ takes a dictionary who must have values of dimensionality (..., A, A)
    and set the off-diagonal (surface) elements to zero
    """
    verify_model_parameters(model)
    for key, value in model.items():
        if hasattr(value, 'shape') and len(value.shape) >= 2:
            # ndims = len(value.shape)
            for a, b in it.permutations(range(model[VMK.A]), 2):
                model[key][..., a, b] = 0.0
            # x = np.diagonal(model[key], axis1=ndims-2, axis2=ndims-1).copy()
            # print(x.shape)
            # model[key][..., :, :] = np.diagonal(model[key], axis1=ndims-2, axis2=ndims-1).copy()
    return


def remove_coupling_from_model(path_source, path_destination):
    """reads in a model from path_source whose values can have dimensionality (..., A, A)
    creates a new model whose values have dimensionality (..., A) from the diagonal of the A dimension of the input model
    saves the new model to the provided path_destination"""
    model = load_model_from_JSON(path_source)

    for key, value in model.items():
        if hasattr(value, 'shape') and len(value.shape) >= 2:
            ndims = len(value.shape)
            model[key] = np.diagonal(model[key], axis1=ndims-2, axis2=ndims-1).copy()

    save_diagonal_model_to_JSON(path_destination, model)
    return


def create_harmonic_model(FS):
    """wrapper function to refresh harmonic model"""
    source = FS.path_vib_model
    dest = FS.path_har_model
    remove_coupling_from_model(source, dest)
    s = "Created harmonic model {:s} by removing coupling from {:s}"
    log.debug(s.format(dest, source))
    return dest


def create_basic_diagonal_model(FS):
    """wrapper function to make the simplest diagonal(sampling) model"""
    source = create_harmonic_model(FS)
    dest = FS.path_rho_model

    if os.path.isfile(dest):
        s = "Sampling model {:s} already exists!"
        log.debug(s.format(dest))

    shutil.copyfile(source, dest)

    s = "Created sampling model {:s} by copying {:s}"
    log.debug(s.format(dest, source))
    return dest


def create_random_orthonormal_matrix(A):
    """returns a orthonormal matrix, just a wrapper for scipy.stats.ortho_group.rvs()"""
    from scipy.stats import ortho_group
    return ortho_group.rvs(A)


def create_orthonormal_matrix_lambda_close_to_identity(order, tuning_parameter):
    """ a wrapper for the function defined in pibronic.vibronic.orthonormal """
    U = orthonormal.create.create_orthonormal_matrix_lambda_close_to_identity(order, tuning_parameter)
    return U


def create_fake_coupled_model(FS, tuning_parameter=0.01, transformation_matrix=None):
    """ take the diagonal coupled model and preform a unitary transformation on it to get a dense matrix

    for a tuning_parameter of 0 the transformation matrix (U) is identity
    the larger the value of tuning_parameter the "farther" away from identity U becomes

    if a numpy array of dim AxA is provided in the transformation_matrix then we won't create a new matrix and instead use the provided one
    note: if transformation_matrix is provided then the tuning_parameter is ignored
    """
    assert os.path.isfile(FS.path_vib_model), "coupled_model file doesn't exist!"

    model = load_model_from_JSON(FS.path_vib_model)
    A, N = _extract_dimensions_from_dictionary(model)

    # we will store the matrix in U
    U = transformation_matrix

    # we now need to obtain a matrix through some means
    if U is None:
        # if no parameter is provided we create a new orthonormal matrix
        U = create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter)
    elif isinstance(U, str) and U == "re-use":
        # if this special flag is provided then we will load a previous orthonormal matrix
        U = np.load(FS.path_ortho_mat)
    elif isinstance(U, np.ndarray):
        # and finally we might be given the orthonormal matrix
        assert np.allclose(U.dot(U.T), np.eye(A)), "matrix is not orthonormal"
    else:
        raise Exception("Your transformation_matrix argument is most likely incorrect?")

    # TODO - should we validate that the model is diagonal and not just symmetric in the surfaces?
    assert model_parameters_are_symmetric_in_surfaces(model)
    assert model_parameters_are_symmetric_in_modes(model)

    for key in filter(model.__contains__, VMK.key_list()):

        # the array of coefficients that we are going to modify
        array = model[key].view()

        # simulating a case statement
        op_switcher = {
                        VMK.E: lambda old: np.einsum('bj,jk,ck->bc', U, old, U),
                        VMK.G1: lambda old: np.einsum('bj,ajk,ck->abc', U, old, U),
                        VMK.G2: lambda old: np.einsum('cj,abjk,dk->abcd', U, old, U),
                        VMK.G3: lambda old: np.einsum('dj,abcjk,ek->abcde', U, old, U),
                        VMK.G4: lambda old: np.einsum('ej,abcdjk,fk->abcdef', U, old, U),
                        }

        # calculate the new Hamiltonian, by preforming the transformation with the matrix U
        new_values = op_switcher[key](array)

        # number of modes in each array
        mode_switcher = {VMK.E: 0, VMK.G1: 1, VMK.G2: 2, VMK.G3: 3, VMK.G4: 4}

        # make sure that the transformation was indeed unitary
        for iters in it.product(*it.repeat(range(N), mode_switcher[key])):
            idx = (*iters, ...)
            assert np.allclose(new_values[idx], U.dot(array[idx].dot(U.T)))

        # overwrite the previous array with the new values
        array[:] = new_values

    # backup the old model, i.e. the "original" model
    shutil.copyfile(FS.path_vib_model,  FS.path_orig_model)

    # overwrite the old model with the new model
    save_model_to_JSON(FS.path_vib_model, model)

    # save the orthogonal matrix in case we need to use it later
    np.save(FS.path_ortho_mat, U)
    return


def main():
    """ currently does nothing """
    return


if (__name__ == "__main__"):
    main()
