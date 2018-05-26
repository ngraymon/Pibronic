"""vibronic_model_io.py should handle the majority of file I/O"""

# system imports
# from pathlib import Path
import itertools as it
# import functools as ft
import subprocess
# import fileinput
import shutil
import json
# import math
import mmap
import sys
import os

# third party imports
import fortranformat as ff  # Fortran format for VIBRON
import numpy as np
# from numpy import newaxis as NEW
from numpy import float64 as F64
from numpy.random import uniform as Uniform
import parse

# local imports
from ..log_conf import log
from .. import constants
from .. import helper
from . import file_structure
from . import file_name
from .vibronic_model_keys import VibronicModelKeys as VMK

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
    """ boolean function that returns true if the provided numpy array is diagonal in the surface dimension
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


def generate_vibronic_model_data(paramDict=None):
    """redo this one but otherwise its fine returns e,w,l,q filled with appropriate values"""
    if paramDict is None:
        # default values
        paramDict = {
                    'frequency_range': [0.02, 0.04],
                    'energy_range': [0.0, 2.0],
                    'quadratic_scaling': 0.08,
                    'linear_scaling': 0.04,
                    'nonadiabatic': True,
                    'numStates': 2,
                    'numModes': 3,
                    }

    # withdraw input values
    nonadiabatic = paramDict['nonadiabatic']
    numStates = paramDict['numStates']
    numModes = paramDict['numModes']

    # ranges
    States = range(numStates)
    Modes = range(numModes)

    # for readability and clarity we use these letters
    size = {
            'N': (numModes),
            'A': (numStates),
            'AA': (numStates, numStates),
            'NAA': (numModes, numStates, numStates),
            'NNAA': (numModes, numModes, numStates, numStates),
            }

    # data
    omega = np.zeros(size['N'])
    energy = np.zeros(size['AA'])
    linear = np.zeros(size['NAA'])
    quadratic = np.zeros(size['NNAA'])

    # readability
    minFreq, maxFreq = paramDict['frequency_range']
    minE, maxE = paramDict['energy_range']

    # generate omega
    omega[:] = np.linspace(minFreq, maxFreq,
                           num=numModes, endpoint=True, dtype=F64)

    # generate energy
    energy[:] = Uniform(minE, maxE, size['AA'])
    # force the energy to be symmetric
    energy[:] = np.tril(energy) + np.tril(energy, k=-1).T

    # calculate the linear displacement
    l_shift = (omega / paramDict['linear_scaling'])

    # generate linear terms
    for i in Modes:
        upTri = Uniform(-l_shift[i], l_shift[i], size['AA'])
        # force the linear terms to be symmetric
        linear[:] = np.tril(upTri) + np.tril(upTri, k=-1).T

    # calculate the quadratic displacement
    q_shift = np.sqrt(np.outer(omega, omega)) / paramDict['quadratic_scaling']

    # generate quadratic terms
    for i in Modes:
        for j in range(i, numModes):
            upTri = Uniform(-q_shift[i,j], q_shift[i,j], size['AA'])
            # force the quadratic terms to be symmetric
            quadratic[i, j, ...] = np.tril(upTri) + np.tril(upTri, k=-1).T
            quadratic[j, i, ...] = np.tril(upTri) + np.tril(upTri, k=-1).T

    # if we are building a harmonic model
    # then zero out all off-diagonal entries
    if not nonadiabatic:
        # energy
        energy[:] = np.diagflat(np.diag(energy))
        # linear terms
        for i in Modes:
            linear[i] = np.diag(np.diag(linear[i]))
        # quadratic terms
        for i, j in it.product(Modes, Modes):
            quadratic[i, j, ...] = np.diag(np.diag(quadratic[i, j, ...]))

    # check for symmetry in surfaces
    assert(np.allclose(energy, energy.transpose(1,0)))
    assert(np.allclose(linear, linear.transpose(0,2,1)))
    assert(np.allclose(quadratic, quadratic.transpose(0,1,3,2)))
    # check for symmetry in modes
    assert(np.allclose(quadratic, quadratic.transpose(1,0,2,3)))

    return energy, omega, linear, quadratic
    # and we are done
    return_dict = {VMK.N: numModes,
                   VMK.A: numStates,
                   VMK.E: energy,
                   VMK.w: omega,
                   VMK.G1: linear,
                   VMK.G2: quadratic,
                   }
    return return_dict


def read_model_h_file(path_file_h):
    """ reads/parses molecule_vibron.h file"""
    path_file_params = path_file_h[:-2] + ".in"

    # declare the arrays used to store the model's paramters
    # all numbers have units of electron volts
    excitation_energies = None
    frequencies = None
    linear_couplings = None
    quadratic_couplings = None

    # the frequencies with units of wavenumbers
    wavenumber_freq = None

    helper.verify_file_exists(path_file_h)
    helper.verify_file_exists(path_file_params)

    def get_number_of_electronic_states(file):
        """x"""
        # skip first line
        file.readline()
        # read in line with electronic states
        line = file.readline()
        # verify that we have the correct line
        targetString = 'Number of electronic states'
        if targetString in line:
            number_of_electronic_states = int(line.split()[0])
            # States = range(number_of_electronic_states)
            log.debug("Electronic states: " + str(number_of_electronic_states))
        else:
            s = "Input file {:s} does not contain {:s}"
            raise Exception(s.format(path_file_h, targetString))

        return number_of_electronic_states

    def get_number_of_normal_modes(file):
        """x"""
        # read in line with symmetric normal modes
        line = file.readline()
        # verify that we have the correct line
        targetString = 'Number of symmetric normal modes'
        if targetString in line:
            number_of_symmetric_modes = int(line.split()[0])
            number_of_normal_modes = number_of_symmetric_modes
            log.debug("Symmetric normal modes: " + str(number_of_symmetric_modes))
            log.debug("Total normal modes: " + str(number_of_normal_modes))
        else:
            s = "Input file {:s} does not contain {:s}"
            raise Exception(s.format(path_file_h, targetString))

        # read in line with non-A1 normal modes
        line = file.readline()
        # verify that we have the correct line
        targetString = 'Number of non-A1 normal modes'
        if targetString in line:
            other_modes = int(line.split()[0])
            number_of_normal_modes += other_modes
            # Modes = range(number_of_normal_modes)
            log.debug("non-A1 normal modes: " + str(other_modes))
            log.debug("Total normal modes: " + str(number_of_normal_modes))
        else:
            s = "Input file {:s} does not contain {:s}"
            raise Exception(s.format(path_file_h, targetString))

        return number_of_normal_modes, number_of_symmetric_modes

    def extract_normal_mode_frequencies(file, frequency_array, symmetric_modes):
        """store output in frequency_array"""
        # skip headers
        file.readline()
        file.readline()

        # read in symmetric normal modes
        line = file.readline()
        frequency_array[0:symmetric_modes] = [float(x) for x in line.split()]
        log.debug(frequency_array)

        # skip header
        file.readline()

        # read in non-A1 normal modes
        line = file.readline()
        frequency_array[symmetric_modes:] = [float(x) for x in line.split()]
        log.debug(frequency_array)
        return

    # set the number of normal modes and electronic surfaces
    # store the normal mode frequencies
    with open(path_file_params, "r") as source_file:
        # we will overwrite these default values
        global numModes, numStates
        global Modes, States

        numStates = get_number_of_electronic_states(source_file)
        States = range(numStates)
        numModes, numSymmetricModes = get_number_of_normal_modes(source_file)
        Modes = range(numModes)

        # for readability and clarity we use these letters
        global size
        size = {
                'N': (numModes),
                'AA': (numStates, numStates),
                'NAA': (numModes, numStates, numStates),
                'NNAA': (numModes, numModes, numStates, numStates),
                }

        # Initialize Variables
        frequencies = np.zeros(size['N'])
        excitation_energies = np.zeros(size['AA'])
        linear_couplings = np.zeros(size['NAA'])
        quadratic_couplings = np.zeros(size['NNAA'])

        extract_normal_mode_frequencies(source_file, frequencies, numSymmetricModes)

        # convert to wavenumers up to 2 decimal places
        wavenumber_freq = np.around(frequencies * 8065.5, decimals=2)
        log.debug(wavenumber_freq)
        #

    def extract_energies(path, memmap, energies):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Reference Hamiltonian'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Gradients of heff along normal modes'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip the header
        memmap.readline()

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        lines = stringData.strip().splitlines()

        # save the reference hamiltonian into the energies array
        for d1 in States:
            energies[d1] = lines[d1].split()
        # print(energies)
        return

    def extract_linear_couplings(path, memmap, coupling_terms):
        """frequencies need to be provided in wavenumbers"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Gradients of heff along normal modes'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Diagonal second order corrections of heff'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        for idx, w in enumerate(frequencies):
            # find the next block of linear coupling_terms terms
            next_block = memmap.find(str(w).encode(encoding="utf-8"), begin, end)
            # error handling
            if next_block == -1:
                s = ("Frequency {:f} in wavenumber_freq did not match any "
                     "frequencies in file {:s} while parsing the "
                     "{:s} region."
                     )
                raise Exception(s.format(w, path, beginString))
            # go there
            memmap.seek(next_block)
            # skip header
            memmap.readline()
            # store each line in the array
            for a in States:
                line = memmap.readline().decode(encoding="utf-8")
                coupling_terms[idx, a, :] = line.split()
            # print(coupling_terms[idx])
        return

    def extract_quadratic_couplings(path, memmap, coupling_terms):
        """frequencies need to be provided in wavenumbers"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Diagonal second order corrections of heff'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Off-diagonal second order corrections heff'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        for idx, w in enumerate(frequencies):
            # find the next block of quadratic coupling terms
            next_block = memmap.find(str(w).encode(encoding="utf-8"), begin, end)
            # error handling
            if next_block == -1:
                s = ("Frequency {:f} in wavenumber_freq did not match any "
                     "frequencies in file {:s} while parsing the "
                     "{:s} region."
                     )
                raise Exception(s.format(w, path, beginString))
            # go there
            memmap.seek(next_block)
            # skip header
            memmap.readline()
            # store each line in the array
            for a in States:
                line = memmap.readline().decode(encoding="utf-8")
                coupling_terms[idx, idx, a, :] = line.split()
            # print(coupling_terms[idx, idx])
        return

    def extract_offdiag_quadratic_couplings(path, memmap, coupling_terms):
        """frequencies need to be provided in wavenumbers"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Off-diagonal second order corrections heff'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Diagonal Cubic corrections of heff'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        for idx1, w1 in enumerate(frequencies):
            for idx2, w2 in enumerate(frequencies[:idx1]):
                # find the next block of quadratic coupling terms
                next_block = memmap.find(str(w1).encode(encoding="utf-8"), memmap.tell(), end)
                # error handling
                if next_block == -1:
                    s = ("Frequency {:f} in wavenumber_freq did not "
                         "match any frequencies in file {:s} "
                         "while parsing the {:s} region."
                         )
                    raise Exception(s.format(w, path, beginString))

                # find the next block of quadratic coupling terms
                next_block = memmap.find(str(w2).encode(encoding="utf-8"), memmap.tell(), end)
                # error handling
                if next_block == -1:
                    s = ("Frequency {:f} in wavenumber_freq did not "
                         "match any frequencies in file {:s} "
                         "while parsing the {:s} region."
                         )
                    raise Exception(s.format(w, path, beginString))

                # go there
                memmap.seek(next_block)
                # skip header
                memmap.readline()

                # store each line in the array
                for a in States:
                    line = memmap.readline().decode(encoding="utf-8")
                    coupling_terms[idx1, idx2, a, :] = line.split()
                # print(coupling_terms[idx1, idx2])
        return

    def extract_cubic_couplings(path, memmap, coupling_terms):
        """not implemented at this time"""
        return

    def extract_quartic_couplings(path, memmap, coupling_terms):
        """not implemented at this time"""
        return

    # store the energy offsets, and all the coupling terms
    with open(path_file_h, "r+b") as source_file:
        # access the file using memory map for efficiency
        with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as mm:
            extract_energies(path_file_h, mm, excitation_energies)
            extract_linear_couplings(path_file_h, mm, wavenumber_freq, linear_couplings)
            # extract_quadratic_couplings(path_file_h, mm, wavenumber_freq, quadratic_couplings)
            # extract_offdiag_quadratic_couplings(path_file_h, mm, wavenumber_freq, quadratic_couplings)
            # extract_cubic_couplings(path_file_h, mm, wavenumber_freq, cubic_couplings)
            # extract_quartic_couplings(path_file_h, mm, wavenumber_freq, quartic_couplings)
        #

    # duplicate the lower triangle values into the upper triangle
    # the couplings are a symmetric matrix
    for a, b in it.product(States, States):
        quadratic_couplings[:, :, a, b] += np.tril(quadratic_couplings[:, :, a, b], k=-1).T

    # check for symmetry in surfaces
    assert(np.allclose(excitation_energies, excitation_energies.transpose(1, 0)))
    assert(np.allclose(linear_couplings, linear_couplings.transpose(0, 2, 1)))
    assert(np.allclose(quadratic_couplings, quadratic_couplings.transpose(0, 1, 3, 2)))
    # check for symmetry in modes
    assert(np.allclose(quadratic_couplings, quadratic_couplings.transpose(1, 0, 2, 3)))

    # and we are done
    return_dict = {VMK.N: numModes,
                   VMK.A: numStates,
                   VMK.E: excitation_energies,
                   VMK.w: frequencies,
                   VMK.G1: linear_couplings,
                   VMK.G2: quadratic_couplings,
                   }
    return return_dict


def read_model_op_file(path_file_op):
    """reads/parses molecule_vibron.op file"""

    # declare the arrays used to store the model's paramters
    # all numbers have units of electron volts
    excitation_energies = None
    frequencies = None
    linear_couplings = None
    quadratic_couplings = None
    cubic_couplings = None
    quartic_couplings = None

    global numStates, numModes, States, Modes, size
    # we will overwrite these default values
    numStates = numModes = States = Modes = size = 0

    helper.verify_file_exists(path_file_op)

    def extract_energies(path, memmap):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Electronic Hamitonian'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Electronic transition moments'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip the header
        helper.readlines(memmap, 3)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        lines = [line for line in stringData.strip().splitlines() if "#" not in line and line is not ""]

        # set the parameters
        global numStates, States
        numStates = len(lines)
        States = range(numStates)

        # save the reference hamiltonian into the energies array
        energies = np.zeros((numStates, numStates))
        for a in States:
            list_of_words = lines[a].split()
            assert list_of_words[0] == f"EH_s{a+1:02}_s{a+1:02}"  # this is a formatted string literal (new in python3.6)
            assert list_of_words[-1] == "ev"
            energies[a, a] = list_of_words[2]
        return energies

    def extract_normal_mode_frequencies(path, memmap):
        """store output in frequency_array"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Frequencies'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Zeropoint energy'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip headers
        helper.readlines(memmap, 2)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        lines = [line for line in stringData.strip().splitlines() if "#" not in line and line is not ""]

        # set the parameters
        global numModes, Modes
        numModes = len(lines)
        Modes = range(numModes)

        # extract the numbers and save them in the frequencies array
        frequencies = np.zeros(numModes)
        for j in Modes:
            list_of_words = lines[j].split()
            assert list_of_words[0] == f"w{j+1:02}"  # this is a formatted string literal (new in python3.6)
            assert list_of_words[-1] == "ev"
            frequencies[j] = list_of_words[2]
        return frequencies

    def extract_linear_couplings(path, memmap, coupling_terms):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Linear Coupling Constants'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Diagonal Quadratic Coupling Constants'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip headers
        helper.readlines(memmap, 2)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        stringData = stringData.strip().replace('=', '').replace(', ev', '')
        lines = [line.split() for line in stringData.splitlines() if "#" not in line and line is not ""]

        p = parse.compile("C1_s{a1:d}_s{a2:d}_v{j:d}")

        for line in lines:
            r = p.parse(line[0])
            index_tuple = (r['j']-1, r['a1']-1, r['a2']-1)
            coupling_terms[index_tuple] = line[1]
        # done
        return

    def extract_quadratic_couplings(path, memmap, coupling_terms):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Diagonal Quadratic Coupling Constants'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Cubic Coupling Constants'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip headers
        helper.readlines(memmap, 2)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        stringData = stringData.strip().replace('=', '').replace(', ev', '')
        lines = [line.split() for line in stringData.splitlines() if "#" not in line and line is not ""]

        p = parse.compile("C2_s{a1:d}s{a2:d}_v{j1:d}v{j2:d}")

        for line in lines:
            r = p.parse(line[0])
            index_tuple = (r['j1']-1, r['j2']-1, r['a1']-1, r['a2']-1)
            coupling_terms[index_tuple] = line[1]
        # done
        return

    def extract_cubic_couplings(path, memmap, coupling_terms):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Cubic Coupling Constants'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Quartic Coupling Constants'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip headers
        helper.readlines(memmap, 2)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        stringData = stringData.strip().replace('=', '').replace(', ev', '')
        lines = [line.split() for line in stringData.splitlines() if "#" not in line and line is not ""]

        p = parse.compile("C3_s{a1:d}_s{a2:d}_v{j:d}")

        for line in lines:
            r = p.parse(line[0])
            index_tuple = (r['j']-1, r['j']-1, r['j']-1, r['a1']-1, r['a2']-1)
            coupling_terms[index_tuple] = line[1]
        # done
        return

    def extract_bicubic_couplings(path, memmap, coupling_terms):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Bi-Cubic Constants'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Bi-Quartic Constants'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip headers
        helper.readlines(memmap, 2)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        stringData = stringData.strip().replace('=', '').replace(', ev', '')
        lines = [line.split() for line in stringData.splitlines() if "#" not in line and line is not ""]

        p = parse.compile("B3_s{a1:d}s{a2:d}_v{j1:d}v{j2:d}")

        for line in lines:
            r = p.parse(line[0])
            # need to confirm this?
            index_tuple = (r['j1']-1, r['j2']-1, r['j2']-1, r['a1']-1, r['a2']-1)
            coupling_terms[index_tuple] = line[1]
        # done
        return

    def extract_quartic_couplings(path, memmap, coupling_terms):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Quartic Coupling Constants'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'Bi-Cubic Constants'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip headers
        helper.readlines(memmap, 2)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        stringData = stringData.strip().replace('=', '').replace(', ev', '')
        lines = [line.split() for line in stringData.splitlines() if "#" not in line and line is not ""]

        p = parse.compile("C4_s{a1:d}_s{a2:d}_v{j:d}")

        for line in lines:
            r = p.parse(line[0])
            index_tuple = (r['j']-1, r['j']-1, r['j']-1, r['j']-1, r['a1']-1, r['a2']-1)
            coupling_terms[index_tuple] = line[1]
        # done
        return

    def extract_biquartic_couplings(path, memmap, coupling_terms):
        """x"""
        # find the begining and ending of the important region
        memmap.seek(0)  # start looking from the begining of the file
        beginString = 'Bi-Quartic Constants'
        begin = helper.find_string_in_file(memmap, path, beginString)
        endString = 'end-parameter-section'
        end = helper.find_string_in_file(memmap, path, endString)

        # go to the begining of that region
        memmap.seek(begin)

        # skip headers
        helper.readlines(memmap, 2)

        # read all the relevant data
        byteData = memmap.read(end - memmap.tell())
        stringData = byteData.decode(encoding="utf-8")
        stringData = stringData.strip().replace('=', '').replace(', ev', '')
        lines = [line.split() for line in stringData.splitlines() if "#" not in line and line is not ""]

        list_B4 = filter(lambda item: item[0].startswith("B4"), lines)
        pB4 = parse.compile("B4_s{a1:d}s{a2:d}_v{j1:d}v{j2:d}")

        list_A4 = filter(lambda item: item[0].startswith("A4"), lines)
        pA4 = parse.compile("A4_s{a1:d}s{a2:d}_v{j1:d}v{j2:d}")

        for line in list_B4:
            r = pB4.parse(line[0])
            index_tuple = (r['j1']-1, r['j2']-1, r['j2']-1, r['j2']-1, r['a1']-1, r['a2']-1)
            coupling_terms[index_tuple] = line[1]

        for line in list_A4:
            r = pA4.parse(line[0])
            index_tuple = (r['j1']-1, r['j1']-1, r['j2']-1, r['j2']-1, r['a1']-1, r['a2']-1)
            coupling_terms[index_tuple] = line[1]
        # done
        return

    # store the energy offsets, and all the coupling terms
    with open(path_file_op, "r+b") as source_file:

        # access the file using memory map for efficiency
        with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as mm:

            frequencies = extract_normal_mode_frequencies(path_file_op, mm)
            excitation_energies = extract_energies(path_file_op, mm)

            # for readability and clarity we use these letters
            A = numStates
            N = numModes
            size = {
                    'N': (N),
                    'AA': (A, A),
                    'NAA': (N, A, A),
                    'NNAA': (N, N, A, A),
                    'NNNAA': (N, N, N, A, A),
                    'NNNNAA': (N, N, N, N, A, A),
                    }
            # size = {
            #         'N': (numModes),
            #         'AA': (numStates, numStates),
            #         'NAA': (numModes, numStates, numStates),
            #         'NNAA': (numModes, numModes, numStates, numStates),
            #         'NNNAA': (numModes, numModes, numModes, numStates, numStates),
            #         'NNNNAA': (numModes, numModes, numModes, numModes, numStates, numStates),
            #         }

            # Assert/Initialize array sizes
            assert frequencies.shape[0] == size['N'], "Incorrect array dimensions"
            assert excitation_energies.shape == size['AA'], "Incorrect array dimensions"

            # predefine these arrays now that we know the values of N and A are
            linear_couplings = np.zeros(size['NAA'])
            quadratic_couplings = np.zeros(size['NNAA'])
            cubic_couplings = np.zeros(size['NNNAA'])
            quartic_couplings = np.zeros(size['NNNNAA'])

            # read in the rest of the parameters
            extract_linear_couplings(path_file_op, mm, linear_couplings)
            extract_quadratic_couplings(path_file_op, mm, quadratic_couplings)
            extract_cubic_couplings(path_file_op, mm, cubic_couplings)
            extract_quartic_couplings(path_file_op, mm, quartic_couplings)
            extract_bicubic_couplings(path_file_op, mm, cubic_couplings)
            extract_biquartic_couplings(path_file_op, mm, quartic_couplings)

    # duplicate the lower triangle values into the upper triangle

    # do we not account for the symmetry in surfaces? this might be incorrect

    linear_couplings[:] += linear_couplings.transpose(0, 2, 1)

    # the couplings are a symmetric matrix
    for a, b in it.product(States, States):
        quadratic_couplings[:, :, a, b] += np.tril(quadratic_couplings[:, :, a, b], k=-1).T
        quadratic_couplings[:, :, a, b] += quadratic_couplings[:, :, b, a]
        # are these two correct?
        cubic_couplings[:, :, :, a, b] += np.tril(cubic_couplings[:, :, :, a, b], k=-1).T
        cubic_couplings[:, :, :, a, b] += cubic_couplings[:, :, :, b, a]  # this is probably incorrect
        quartic_couplings[:, :, :, :, a, b] += np.tril(quartic_couplings[:, :, :, :, a, b], k=-1).T
        quartic_couplings[:, :, :, :, a, b] += quartic_couplings[:, :, :, :, b, a]

    # don't overcount the diagonals
    for a in States:
        linear_couplings[:, a, a] /= 2.
        quadratic_couplings[:, :, a, b] /= 2.
        cubic_couplings[:, :, :, a, b] /= 2.
        quartic_couplings[:, :, :, :, a, b] /= 2.

    # check for symmetry in surfaces
    assert np.allclose(excitation_energies, excitation_energies.transpose(1, 0))
    assert np.allclose(linear_couplings, linear_couplings.transpose(0, 2, 1))
    assert np.allclose(quadratic_couplings, quadratic_couplings.transpose(0, 1, 3, 2))
    assert np.allclose(cubic_couplings, cubic_couplings.transpose(0, 1, 2, 4, 3))
    assert np.allclose(quartic_couplings, quartic_couplings.transpose(0, 1, 2, 3, 5, 4))

    # check for symmetry in modes
    assert np.allclose(quadratic_couplings, quadratic_couplings.transpose(1, 0, 2, 3))
    # either (q^3) or (q, q^2)
    assert np.allclose(cubic_couplings, cubic_couplings.transpose(0, 2, 1, 3, 4))
    # either (q, q^3) or (q^2, q^2)
    assert np.allclose(quartic_couplings, quartic_couplings.transpose(0, 1, 3, 2, 4, 5))

    # and we are done
    maximal_dict = {VMK.N: numModes,
                    VMK.A: numStates,
                    VMK.E: excitation_energies,
                    VMK.w: frequencies,
                    VMK.G1: linear_couplings,
                    VMK.G2: quadratic_couplings,
                    VMK.G3: cubic_couplings,
                    VMK.G4: quartic_couplings,
                    }

    # if the arrays only have zeros then we might not need to store them?
    return_dict = dict((k, v) for k, v in maximal_dict.items() if not np.all(v == 0))

    return return_dict


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
    except:
        # The url wasn't valid
        raise Exception("Incorrect https link {:s}".format(url))


def read_model_auto_file(filename):
    """
    if the shift to MCTDH file structure is permananent this function will no longer be needed

    Read Vibronic Model file (cp.auto) which
    contains all information on the approximate
    Born-Oppenheimer Ground State PES.

    Returns:
    number of electronic states
    number of normal modes

    Excitation energies: (nel, nel)
    frequencies (nmode)
    linear couplings: (nmode, nel, nel)
    quadratic couplings: (nmode, nmode, nel, nel)
    """
    nel = 2  # Number of minima [energy wells]
    nmode = 12  # Number of normal modes
    mode_range = range(nmode)
    el_range = range(nel)

    # Read file (Iterator object)
    file_object = open(filename, 'r')

    # first lets strip out all the blank lines and lines like ------
    lines = [line.rstrip() for line in file_object.readlines() if ((len(line.strip()) != 0) and ("--------" not in line.strip()))]
    f_iter = iter(lines)  # make an iterator

    # =====
    # Title
    # =====

    title_header = ff.FortranRecordReader('(A50, A20)')
    title_header.read(next(f_iter))

    # ==========================
    # 0. Header and Normal modes
    # ==========================

    numgrid = ff.FortranRecordReader('(A26, i4)').read(next(f_iter))[1]      # read the type of grid in ACESII
    num_irreps = ff.FortranRecordReader('(A36, i4)').read(next(f_iter))[1]   # number of vibrational irreps
    nmodes = ff.FortranRecordReader('(A30, i4)').read(next(f_iter))[1]       # read the total number of modes per symmetry

    # Initialize Variables
    excitation_energies = np.zeros((nel, nel))
    frequencies         = np.zeros((nmode))
    linear_couplings    = np.zeros((nmode, nel, nel))
    quadratic_couplings = np.zeros((nmode, nmode, nel, nel))

    # we can either grab the frequencies from the ref_freq_data, or the parent hessian
    if(False):
        # the frequency given here is in cm-1
        frequencies[:] = np.asarray(ff.FortranRecordReader('(A31, 12F8.2)').read(next(f_iter))[1:], dtype=F64)
    else:
        next(f_iter)

    # ==================
    # 1. Parent Gradient
    # ==================

    title_header.read(next(f_iter))
    gradient_header_parent = ff.FortranRecordReader('(t3, A17, i4, t22, A15, E16.10)')
    for i in mode_range:
        next(f_iter)

    # =================
    # 2. Parent Hessian
    # =================

    title_header.read(next(f_iter))
    hessian_header_parent = ff.FortranRecordReader('(t3, A17, i4, t22, A15, F8.2, A5, F12.6, A5)')
    try:
        for i in mode_range:
            # store the value incase exception occurs
            temp_str = next(f_iter)
            frequencies[i] = float(temp_str.partition("cm-1")[2].partition("eV")[0])
    except ValueError as err_obj:
        print("Having issues parsing the Hessian parent energy, why are the frequences not digits?")
        print(temp_str)
        print(temp_str.partition("cm-1")[2])
        print(temp_str.partition("cm-1")[2].partition("eV")[0])
        raise err_obj

    # ========================
    # 3. Reference Hamiltonian
    # ========================

    title_header.read(next(f_iter))
    energy_header = ff.FortranRecordReader('2(F12.6)')
    for a in el_range:
        excitation_energies[a,:] = energy_header.read(next(f_iter))

    # ===============================
    # 4. Linear couplings (gradients)
    # ===============================

    title_header.read(next(f_iter))
    gradient_header = ff.FortranRecordReader('(t3, A15, i4, A15, F10.2)')
    for i in mode_range:
        gradient_header.read(next(f_iter))  # should 'read in' the header
        for a in el_range:
            linear_couplings[i, a, :] = energy_header.read(next(f_iter))

    # ========================================================
    # 5. Quadratic couplings [Diagonal corrections of Hessian]
    # ========================================================
    #

    title_header.read(next(f_iter))
    for i in mode_range:
        gradient_header.read(next(f_iter))
        for a in el_range:
            quadratic_couplings[i, i, a, :] = energy_header.read(next(f_iter))

    # ============================================================
    # 6. Quadratic couplings [Off-diagonal corrections of Hessian]
    # ============================================================

    title_header.read(next(f_iter))
    for i in mode_range:
        for j in range(i):
            next(f_iter)  # 'reading in' the first line/title
            for a in el_range:
                quadratic_couplings[i, j, a, :] = energy_header.read(next(f_iter))

    for a in el_range:
        for b in el_range:
            quadratic_couplings[:,:,a,b] += np.tril(quadratic_couplings[:,:,a,b], k=-1).T

    # check for symmetry in surfaces
    for a in el_range:
        for b in el_range:
            assert(np.allclose(linear_couplings[:,a,b], linear_couplings[:,b,a]))
            assert(np.allclose(quadratic_couplings[:,:,a,b], quadratic_couplings[:,:,b,a]))

    # check for symmetry in modes
    for i in mode_range:
        for j in mode_range:
            assert(np.allclose(quadratic_couplings[i,j,:,:], quadratic_couplings[j,i,:,:]))

    return nel, nmode, excitation_energies, frequencies, linear_couplings, quadratic_couplings


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
    """ take the diagonal coupled model and preform a unitary transformation on it to get a dense matrix """
    assert os.path.isfile(FS.path_vib_model), "coupled_model file doesn't exist!"

    model_dict = load_model_from_JSON(FS.path_vib_model)
    A, N = _extract_dimensions_from_dictionary(model_dict)
    U = create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter)
    # print("U ", U)

    # TODO - all the for loops are just hard checks, they should be factored out into test cases
    # in the long term

    for key in VMK.key_list():
        if key not in model_dict:
            continue

        array = model_dict[key].view()
        assert model_array_diagonal_in_surfaces(array), f"{key} not diagonal in surfaces"

        if key is VMK.E:
            new_values = np.einsum('bj,jk,ck->bc', U, array, U)

        elif key is VMK.G1:
            new_values = np.einsum('bj,ajk,ck->abc', U, array, U)
            for j in range(N):
                assert np.allclose(new_values[j, ...], U.dot(array[j, ...].dot(U.T)))

        elif key is VMK.G2:
            new_values = np.einsum('cj,abjk,dk->abcd', U, array, U)
            for j1, j2, in zip(range(N), range(N)):
                assert np.allclose(new_values[j1, j2, ...], U.dot(array[j1, j2, ...].dot(U.T)))

        elif key is VMK.G3:
            new_values = np.einsum('dj,abcjk,ek->abcde', U, array, U)
            for j1, j2, j3 in zip(range(N), range(N), range(N)):
                assert np.allclose(new_values[j1, j2, j3, ...],
                                   U.dot(array[j1, j2, j3, ...].dot(U.T)))

        elif key is VMK.G4:
            new_values = np.einsum('ej,abcdjk,fk->abcdef', U, array, U)
            for j1, j2, j3, j4 in zip(range(N), range(N), range(N), range(N)):
                assert np.allclose(new_values[j1, j2, j3, j4, ...],
                                   U.dot(array[j1, j2, j3, j4, ...].dot(U.T)))

        # overwrite the previous array with the new values
        array[:] = new_values

    # now we need to backup the old model and save the new one
    source = FS.path_vib_model
    dest = FS.path_vib_params + file_name.original_model
    # backup the old model
    shutil.copyfile(source, dest)
    # save the new one
    save_model_to_JSON(FS.path_vib_model, model_dict)

    # save the Orthogonal matrix in case we need to use it later
    np.save(FS.path_vib_params + "orthogonal_matrix", U)
    return


if (__name__ == "__main__"):
    print("Currently does nothing")
