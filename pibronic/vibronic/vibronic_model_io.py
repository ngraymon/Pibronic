"""vibronic_model_io.py should handle the majority of file I/O"""

# system imports
import itertools as it
import hashlib
import shutil
import json
import os
from os.path import join

# third party imports
import numpy as np
from numpy import float64 as F64
from numpy.random import uniform as Uniform

# local imports
from ..log_conf import log
from .. import constants
from ..data import file_structure
from ..data import file_name
from .vibronic_model_keys import VibronicModelKeys as VMK
from . import model_auto
from . import model_op
from . import model_h

np.set_printoptions(precision=8, suppress=True)  # Print Precision!

# TODO - made the design decision that if a key is not present in the json file that implies all the values are zero
# - need to make sure this is enforced across all the code


def model_shape_dict(A, N):
    """ returns a dictionary with the same keys as the .json file whose values are tuples representing the dimensonality of the associated value in the .json file
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


def sample_shape_dict(A, N):
    """ returns a dictionary with the same keys as the .json file whose values are tuples representing the dimensonality of the associated value in the .json file
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


def model_array_diagonal_in_surfaces(array):
    """ boolean function that returns true if the provided numpy array is diagonal or symmetric in the surface dimension
    where the surface dimensions (A) are by convention the last two dimensions
    this function assumes that the array is properly formatted

    """

    new_dims = list(range(array.ndim))
    # swap the last two dimensions, which by convention are the surface dimensions
    new_dims[-1], new_dims[-2] = new_dims[-2], new_dims[-1]

    return np.allclose(array, array.transpose(new_dims))


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


def verify_sample_parameters(kwargs):
    """make sure the provided sample parameters follow the file conventions"""
    assert VMK.N in kwargs, "need the number of modes"
    assert VMK.A in kwargs, "need the number of surfaces"

    A, N = _extract_dimensions_from_dictionary(kwargs)
    shape_dict = sample_shape_dict(A, N)

    for key, value in kwargs.items():
        if key in shape_dict:
            assert kwargs[key].shape == shape_dict[key], f"{key} have incorrect shape"
        else:
            log.debug(f"Found key {key} which is not present in the default dictionary")

    return


def _hash(string):
    """creates a sha512 hash of the input string and returns the byte representation"""
    m = hashlib.sha512()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


def create_model_hash(FS=None, path=None):
    """ create a has of the coupled_model.json file's contents
    this is used to confirm that result files were generated for the current model and not an older one
    uses a FileStructure or an absolute path to the file"""
    if FS is None:
        assert path is not None, "no arguments provided"
    else:
        path = FS.path_vib_model

    with open(path, mode='r', encoding='utf8') as file:
        string = file.read()

    return _hash(string)


def create_sampling_hash(FS=None, path=None):
    """ create a has of the sampling_model.json file's contents
    this is used to confirm that result files were generated for the current model and not an older one
    uses a FileStructure or an absolute path to the file"""
    if FS is None:
        assert path is not None, "no arguments provided"
    else:
        path = FS.path_rho_model

    with open(path, mode='r', encoding='utf8') as file:
        string = file.read()

    return _hash(string)


def generate_default_root():
    """backwards compatibility fix for functions that rely on having a predefined default_root"""
    return "/work/ngraymon/pimc/"


def pretty_print_model(id_data, unitsOfeV=False):
    """one method of printing the models in a human readable format - outdated and should probably be removed or modified"""

    # import pandas as pd
    from xarray import DataArray as dArr

    path_root = generate_default_root()
    FS = file_structure.FileStructure(path_root, id_data)
    kwargs = load_model_from_JSON(FS.path_vib_model)

    # parameter values
    A = kwargs[VMK.A]
    N = kwargs[VMK.N]
    States = range(A)
    Modes = range(N)

    # formatting
    a_labels = ['a%d' % a for a in range(1, A+1)]
    b_labels = ['b%d' % a for a in range(1, A+1)]
    i_labels = ['i%d' % j for j in range(1, N+1)]
    j_labels = ['j%d' % j for j in range(1, N+1)]

    # label the arrays for readability
    energy = kwargs[VMK.E].view()
    omega = kwargs[VMK.w].view()
    linear = kwargs[VMK.G1].view()
    quadratic = kwargs[VMK.G2].view()

    # by default convert the output to wavenumbers
    conversionFactor = 1 if unitsOfeV else constants.wavenumber_per_eV

    # stringify the lists
    for i in Modes:
        omega[i] *= conversionFactor
        omega[i] = str(omega[i])

    for a, b in it.product(States, States):
        energy[a][b] *= conversionFactor
        energy[a][b] = str(energy[a][b])

    for a, b, i in it.product(States, States, Modes):
        linear[i][a][b] *= conversionFactor
        linear[i][a][b] = str(linear[i][a][b])

    for a, b, i, j in it.product(States, States, Modes, Modes):
        quadratic[i][j][a][b] *= conversionFactor
        quadratic[i][j][a][b] = str(quadratic[i][j][a][b])

    # load the data into xarray's DataArrays
    omegaArray = dArr(omega, name=VMK.w,
                      coords=[i_labels],
                      dims=['mode i'],)

    energyArray = dArr(energy, name=VMK.E,
                       coords=[a_labels, b_labels],
                       dims=['surface a', 'surface b'],)

    linArray = dArr(linear, name=VMK.G1,
                    coords=[i_labels, a_labels, b_labels],
                    dims=['mode i', 'surface a', 'surface b'],)

    quadArray = dArr(quadratic, name=VMK.G2,
                     coords=[i_labels, j_labels, a_labels, b_labels],
                     dims=['mode i', 'mode j ', 'surface a', 'surface b'],)

    # print the data, relying on panda's DataArrays to printin a human legible manner
    print(omegaArray.to_dataframe(),
          energyArray.to_dataframe(),
          linArray.to_dataframe(),
          quadArray.to_dataframe(),
          sep="\n",
          )

    return


def _generate_linear_terms(linear_terms, shape, displacement):
    """ generate linear terms that are 'reasonable' """
    for i in range(shape[VMK.w]):
        upTri = Uniform(-displacement[i], displacement[i], shape[VMK.E])
        # force the linear terms to be symmetric
        linear_terms[:] = np.tril(upTri) + np.tril(upTri, k=-1).T
    return


def _generate_quadratic_terms(quadratic_terms, shape, displacement, N):
    """ generate quadratic terms that are 'reasonable' """
    for i, j in it.combinations_with_replacement(range(N), 2):
        upTri = Uniform(-displacement[i, j], displacement[i, j], shape[VMK.E])
        # force the quadratic terms to be symmetric
        quadratic_terms[i, j, ...] = np.tril(upTri) + np.tril(upTri, k=-1).T
        quadratic_terms[j, i, ...] = np.tril(upTri) + np.tril(upTri, k=-1).T
    return


def _remove_coupling_from_generated_model(modes, energy, linear_terms, quadratic_terms):
    """ x """
    energy[:] = np.diagflat(np.diag(energy))
    for i in modes:
        linear_terms[i] = np.diag(np.diag(linear_terms[i]))
    for i, j in it.product(modes, modes):
        quadratic_terms[i, j, ...] = np.diag(np.diag(quadratic_terms[i, j, ...]))
    return


def generate_vibronic_model_data(paramDict=None):
    """redo this one but otherwise its fine returns e,w,l,q filled with appropriate values"""
    if paramDict is None:
        paramDict = {   # default values
                        'frequency_range': [0.02, 0.04],
                        'energy_range': [0.0, 2.0],
                        'quadratic_scaling': 0.08,
                        'linear_scaling': 0.04,
                        'nonadiabatic': True,
                        'numStates': 2,
                        'numModes': 3,
                        }
    # readability
    minE, maxE = paramDict['energy_range']
    minFreq, maxFreq = paramDict['frequency_range']

    # ranges for convenience
    numModes = paramDict['numModes']
    # numStates = paramDict['numStates']
    Modes = range(numModes)
    # States = range(numStates)

    # generate the array dimensions
    shape = model_shape_dict(paramDict['numStates'], numModes)

    # initialize arrays
    frequencies = np.zeros(shape[VMK.w])
    energy = np.zeros(shape[VMK.E])
    linear_terms = np.zeros(shape[VMK.G1])
    quadratic_terms = np.zeros(shape[VMK.G2])

    # generate frequencies
    frequencies[:] = np.linspace(minFreq, maxFreq, num=numModes, endpoint=True, dtype=F64)

    # generate energy
    energy[:] = Uniform(minE, maxE, shape[VMK.E])
    # force the energy to be symmetric
    energy[:] = np.tril(energy) + np.tril(energy, k=-1).T

    # calculate the linear displacement
    l_shift = frequencies / paramDict['linear_scaling']
    _generate_linear_terms(linear_terms, shape, l_shift)

    # calculate the quadratic displacement
    q_shift = np.sqrt(np.outer(frequencies, frequencies)) / paramDict['quadratic_scaling']
    _generate_quadratic_terms(quadratic_terms, shape, q_shift, Modes)

    # if we are building a harmonic model then zero out all off-diagonal entries
    if not paramDict['nonadiabatic']:
        _remove_coupling_from_generated_model(Modes, energy, linear_terms, quadratic_terms)

    assert model_array_diagonal_in_surfaces(energy), "energy not symmetric in surfaces"
    assert model_array_diagonal_in_surfaces(linear_terms), "linear_terms not symmetric in surfaces"
    assert model_array_diagonal_in_surfaces(quadratic_terms), "quadratic_terms not symmetric in surfaces"
    assert np.allclose(quadratic_terms, quadratic_terms.transpose(1, 0, 2, 3)), "quadratic_terms not symmetric in surfaces"

    # TODO - which of these returns is the correct one to use?
    # # and we are done
    # return_dict = {VMK.N: numModes,
    #                VMK.A: numStates,
    #                VMK.E: energy,
    #                VMK.w: frequencies,
    #                VMK.G1: linear_terms,
    #                VMK.G2: quadratic_terms,
    #                }
    # return return_dict

    return energy, frequencies, linear_terms, quadratic_terms


def read_model_h_file(path_file_h):
    """ wrapper function to maintain functionality - possible remove in the future"""
    model_h.read_model_h_file(path_file_h)
    return


def read_model_op_file(path_file_op):
    """ wrapper function to maintain functionality - possible remove in the future"""
    model_op.read_model_op_file(path_file_op)
    return


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
        # The url wasn't valid
        raise Exception("Incorrect https link {:s}".format(url))


def read_model_auto_file(path_file_auto):
    """ wrapper function to maintain functionality - possible remove in the future"""
    model_auto.read_model_auto_file(path_file_auto)
    return


def _extract_dimensions_from_dictionary(dictionary):
    """x"""
    N = int(dictionary[VMK.N])
    A = int(dictionary[VMK.A])
    return A, N


def _extract_dimensions_from_file(path):
    """x"""
    assert os.path.isfile(path), f"invalid path:\n{path:s}"
    with open(path, mode='r', encoding='UTF8') as file:
        input_dictionary = json.loads(file.read())
    VMK.change_dictionary_keys_from_strings_to_enum_members(input_dictionary)
    return _extract_dimensions_from_dictionary(input_dictionary)


def extract_dimensions_of_coupled_model(FS=None, path=None):
    """return number_of_modes and number_of_surfaces for coupling_model.json files by using a FileStructure or an absolute path to the file"""
    if FS is None:
        assert path is not None, "no arguments provided"
    else:
        path = FS.path_vib_model
    return _extract_dimensions_from_file(path)


def extract_dimensions_of_sampling_model(FS=None, path=None):
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
    VMK.change_dictionary_keys_from_enum_members_to_strings(dictionary)
    """ converts each numpy array to a list so that json can serialize them properly"""
    for key, value in list(dictionary.items()):
        if isinstance(value, (np.ndarray, np.generic)):
            if np.count_nonzero(value) > 0:
                dictionary[key] = value.tolist()
            else:
                del dictionary[key]
        else:
            log.debug(f"Value {value} with Key {key} does not appear to be an ndarray")

    with open(path, mode='w', encoding='UTF8') as target_file:
        target_file.write(json.dumps(dictionary))

    return


def save_model_to_JSON(path, dictionary):
    """ wrapper for _save_to_JSON
    calls verify_model_parameters() before calling _save_to_JSON()
    """
    verify_model_parameters(dictionary)
    log.debug(f"Saving model to {path:s}")
    _save_to_JSON(path, dictionary)
    return


def save_sample_to_JSON(path, dictionary):
    """ wrapper for _save_to_JSON
    calls verify_sample_parameters() before calling _save_to_JSON()
    """
    verify_sample_parameters(dictionary)
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

    # no arrays were provided so return newly created arrays after filling them with the approriate values
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


def load_sample_from_JSON(path, dictionary=None):
    """
    if kwargs is not provided then returns a dictionary filled with the values stored in the .json file located at path

    if kwargs is provided then all values are overwritten (in place) with the values stored in the .json file located at path
    """
    log.debug(f"Loading rho model (sampling model) from {path:s}")

    # no arrays were provided so return newly created arrays after filling them with the approriate values
    if not bool(dictionary):
        new_model_dict = _load_from_JSON(path)

        # TODO - we might want to make sure that none of the values in the dictionary have all zero values or are None

        verify_sample_parameters(new_model_dict)
        return new_model_dict

    # arrays were provided so fill them with the appropriate values
    else:
        verify_sample_parameters(dictionary)
        _load_inplace_from_JSON(path, dictionary)
        # check twice? might as well be cautious for the moment until test cases are written
        verify_sample_parameters(dictionary)
    return
# ------------------------------------------------------------------------


def remove_coupling_from_model(path_source, path_destination):
    """reads in a model from path_source whose values can have dimensionality (..., A, A)
    creates a new model whose values have dimensionality (..., A) from the diagonal of the A dimension of the input model
    saves the new model to the provided path_destination"""
    kwargs = load_model_from_JSON(path_source)

    for key, value in kwargs.items():
        if hasattr(value, 'shape') and len(value.shape) >= 2:
            ndims = len(value.shape)
            kwargs[key] = np.diagonal(kwargs[key], axis1=ndims-2, axis2=ndims-1).copy()
    save_sample_to_JSON(path_destination, kwargs)
    return


def create_harmonic_model(FS):
    """wrapper function to refresh harmonic model"""
    source = FS.path_vib_params + file_name.coupled_model
    dest = FS.path_vib_params + file_name.harmonic_model
    remove_coupling_from_model(source, dest)
    s = "Created harmonic model {:s} by removing coupling from {:s}"
    log.debug(s.format(dest, source))
    return dest


def create_basic_sampling_model(FS):
    """wrapper function to make the simplest sampling model"""
    source = create_harmonic_model(FS)
    dest = FS.path_rho_params + file_name.sampling_model

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


def create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter):
    """returns a orthonormal matrix which is identity if lambda is 0
    the larger the value of lambda the 'farther' away the matrix is from identity
    takes: number of surfaces and lambda value
    """
    from scipy.linalg import expm

    # TODO - this function should probably be in another module that provides general math
    # functions for all modules

    # scale the tuning parameter by the size of the matrix
    tuning_parameter /= A
    # create a random matrix
    rand_matrix = np.random.rand(A, A)
    # generate a skew symmetric matrix
    skew_matrix = rand_matrix - rand_matrix.T
    # generate an orthonormal matrix, which depends on the tuning parameter
    ortho_matrix = expm(tuning_parameter * skew_matrix)
    assert np.allclose(ortho_matrix.dot(ortho_matrix.T), np.eye(A)), "matrix is not orthonormal"
    return ortho_matrix


def create_fake_coupled_model(FS, tuning_parameter=0.01):
    """ take the diagonal coupled model and preform a unitary transformation on it to get a dense matrix
    for a tuning_parameter of 0 the transformation matrix (U) is identity
    the larger the value of tuning_parameter the "farther" away from identity U becomes
    """
    assert os.path.isfile(FS.path_vib_model), "coupled_model file doesn't exist!"

    model_dict = load_model_from_JSON(FS.path_vib_model)
    A, N = _extract_dimensions_from_dictionary(model_dict)
    U = create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter)

    for key in filter(model_dict.__contains__, VMK.key_list()):

        # the array of coefficients that we are going to modify
        array = model_dict[key].view()

        # TODO - do we really need to assert this? should this already have been called when we loaded the model?
        assert model_array_diagonal_in_surfaces(array), f"{key} not diagonal in surfaces"
        """ we assert if the array is diagonal in modes inside each if statement because the checks are all unique, the surface check is the same because all arrays have the same structure where the last two dimensions are the surface dimensions """

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

    # now we need to backup the old model and save the new one
    source = FS.path_vib_model
    dest = FS.path_vib_params + file_name.original_model
    # backup the old model
    shutil.copyfile(source, dest)
    # save the new one
    save_model_to_JSON(source, model_dict)

    # save the orthogonal matrix in case we need to use it later
    np.save(join(FS.path_vib_params, "orthogonal_matrix"), U)
    return


if (__name__ == "__main__"):
    print("Currently does nothing")
