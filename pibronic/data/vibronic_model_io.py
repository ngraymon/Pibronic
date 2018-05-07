# vibronic_model_io.py
# should handle the majority of file I/O

# system imports
from pathlib import Path
import itertools as it
import functools as ft
import subprocess
import fileinput
import shutil
import json
import math
import mmap
import sys
import os

# third party imports
import fortranformat as ff  # Fortran format for VIBRON
import numpy as np
from numpy import newaxis as NEW
from numpy import float64 as F64
from numpy.random import uniform as Uniform
import parse

# local imports
from ..log_conf import log
from .. import constants
from .. import helper
from . import file_structure


# this function should most likely be removed
def checkOS():
    """define OS dependent paths to files"""
    global kernel, path_default_root, dir_workspace

    if sys.platform.startswith('linux'):
        kernel = "linux"
        # identify between different versions of linux
        proc = subprocess.Popen('hostid', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        hostid = proc.stdout.read().decode()
        if hostid[0:4] == "007f":
            path_default_root = "/home/neil/Desktop/pimc/"
            # dir_workspace = "~"
        elif hostid[0:4] == "360a":
            path_default_root = "/work/ngraymon/pimc/"
            # dir_workspace = "~""
    elif sys.platform.startswith('darwin'):
        kernel = "darwin"
        # path_default_root = "~"
        # dir_workspace = "~"
    else:
        raise OSError("not linux or OSX, problem with filestructure")
    return


# Print Precision!
np.set_printoptions(precision=8, suppress=True)

# by default the following name is used for the json file
# json_filename = "coupled_model.json"


# Number of minima [energy wells]
nel = 2
# Number of normal modes
nmode = 12  # default value

mode_range = range(nmode)
el_range = range(nel)


dir_vib = "data_set_{:d}/"
dir_rho = "rho_{:d}/"

# the names of the sub directories
list_sub_dirs = [
    "parameters/",
    "results/",
    "execution_output/",
    "plots/",
    ]


def pretty_print_model(id_model, unitsOfeV=False):
    """one method of printing the models in a human readable format"""
    # checkOS()
    import pandas as pd
    from xarray import DataArray as dArr

    # parameter values
    N, A = get_nmode_nsurf_from_coupled_model(id_model)
    numStates = A
    numModes = N
    States = range(numStates)
    Modes = range(numModes)

    # formatting
    a_labels = ['a%d' % a for a in range(1, A+1)]
    b_labels = ['b%d' % a for a in range(1, A+1)]
    i_labels = ['i%d' % j for j in range(1, N+1)]
    j_labels = ['j%d' % j for j in range(1, N+1)]

    # load the data
    path = path_default_root + dir_vib + "parameters/" + "coupled_model.json"
    path = path.format(id_model)
    target_file = open(path, mode='r', encoding='UTF8')
    input_dictionary = json.loads(target_file.read())

    # isolate the lists
    energy = input_dictionary["energies"]
    omega = input_dictionary["frequencies"]
    linear = input_dictionary["linear couplings"]
    quadratic = input_dictionary["quadratic couplings"]

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
    omegaArray = dArr(omega,
                      coords=[i_labels],
                      dims=['mode i'],
                      name="Frequencies"
                      )

    energyArray = dArr(energy,
                       coords=[a_labels, b_labels],
                       dims=['surface a', 'surface b'],
                       name="Energies"
                       )

    linArray = dArr(linear,
                    coords=[i_labels, a_labels, b_labels],
                    dims=['mode i', 'surface a', 'surface b'],
                    name="linear terms"
                    )

    quadArray = dArr(quadratic,
                     coords=[i_labels, j_labels, a_labels, b_labels],
                     dims=['mode i', 'mode j ', 'surface a', 'surface b'],
                     name="quadratic terms"
                     )

    # print the data, relying on panda's DataArrays
    # to printin a human legible manner
    print(omegaArray.to_dataframe(),
          energyArray.to_dataframe(),
          linArray.to_dataframe(),
          quadArray.to_dataframe(),
          sep="\n")

    return


# this function should probably be removed - no longer needed
def parse_model_params(path_full):
    """this one could probably be removed? parses model_parameters_source.txt"""
    # does the file exist?
    helper.verify_file_exists(path_full)

    parameter_dictionary = {
        "number_of_surfaces": None,
        "number_of_modes": None,
        "energy_range": None,
        "frequency_range": None,
        "quadratic_scaling": None,
        "linear_scaling": None,
        "temperature_list": None,
        "sample_list": None,
        "bead_list": None,
        }

    with open(path_full, 'r') as source_file:
        while not source_file.readline() == "":
            header = source_file.readline().strip()
            data = source_file.readline().strip()

            if(    header == "quadratic_scaling"
                or header == "linear_scaling"
                ):
                parameter_dictionary[header] = float(data)

            elif(  header == "number_of_surfaces"
                or header == "number_of_modes"
                ):
                parameter_dictionary[header] = int(data)

            elif(  header == "temperature_list"
                or header == "memory_list"
                or header == "sample_list"
                or header == "bead_list"
                ):
                parameter_dictionary[header] = np.fromstring(data, sep=',').astype(int)

            elif(  header == "frequency_range"
                or header == "energy_range"
                ):
                parameter_dictionary[header] = np.fromstring(data, dtype=F64, sep=',')
            else:
                raise ValueError("header {:} is not valid\n"
                                 "Check that your model_parameters_source.txt"
                                 "has the correct formatting".format(header)
                                 )
    return parameter_dictionary


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
    return_dict = {"number of modes": numModes,
                   "number of surfaces": numStates,
                   "energies": energy,
                   "frequencies": omega,
                   "linear couplings": linear,
                   "quadratic couplings": quadratic,
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
        """"""
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
        """"""
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
        """a"""
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
    return_dict = {"number of modes": numModes,
                   "number of surfaces": numStates,
                   "energies": excitation_energies,
                   "frequencies": frequencies,
                   "linear couplings": linear_couplings,
                   "quadratic couplings": quadratic_couplings,
                   }
    return return_dict


def read_model_op_file(path_file_op):
    """ reads/parses molecule_vibron.op file"""

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

    # the couplings are a symmetric matrix
    for a, b in it.product(States, States):
        linear_couplings[:, a, b] += linear_couplings[:, b, a]
        quadratic_couplings[:, :, a, b] += np.tril(quadratic_couplings[:, :, a, b], k=-1).T
        quadratic_couplings[:, :, a, b] += quadratic_couplings[:, :, b, a]
        # are these two correct?
        cubic_couplings[:, :, :, a, b] += np.tril(cubic_couplings[:, :, :, a, b], k=-1).T
        cubic_couplings[:, :, :, a, b] += cubic_couplings[:, :, :, b, a]
        quartic_couplings[:, :, :, :, a, b] += np.tril(quartic_couplings[:, :, :, :, a, b], k=-1).T
        quartic_couplings[:, :, :, :, a, b] += quartic_couplings[:, :, :, :, b, a]

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
    return_dict = {"number of modes": numModes,
                   "number of surfaces": numStates,
                   "energies": excitation_energies,
                   "frequencies": frequencies,
                   "linear couplings": linear_couplings,
                   "quadratic couplings": quadratic_couplings,
                   "cubic couplings": cubic_couplings,
                   "quartic couplings": quartic_couplings,
                   }
    return return_dict


def create_coupling_from_h_file(FS, path_file_h):
    """assumes that the path_file_h is in electronic_structure"""
    # A, N, E, w, l, q = read_model_h_file(path_file_h)
    model_dict = read_model_h_file(path_file_h)
    save_model_to_JSON(FS.path_vib_model, **model_dict)
    log.debug("Created \n{:s}\nfrom\n{:s}\n".format(FS.path_vib_model, path_file_h))
    return FS.path_vib_model


def create_coupling_from_op_file(FS, path_file_op):
    """assumes that the path_file_op is in electronic_structure"""
    # A, N, E, w, l, q = read_model_h_file(path_file_op)
    model_dict = read_model_op_file(path_file_op)
    save_model_to_JSON(FS.path_vib_model, **model_dict)
    log.debug("Created \n{:s}\nfrom\n{:s}\n".format(FS.path_vib_model, path_file_op))
    return FS.path_vib_model


# not working ATM
def create_coupling_from_op_hyperlink(FS, url):
    """assumes that the path_file_op is in electronic_structure"""
    import urllib

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
    '''
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
    '''

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


def save_model_to_JSON(path_full, **kwargs):
    """x"""

    # base case
    assert "number of modes" in kwargs, "need the number of modes"
    assert "number of surfaces" in kwargs, "need the number of surfaces"
    assert "energies" in kwargs, "no energies provided"
    assert "frequencies" in kwargs, "no frequencies provided"
    assert kwargs["energies"].shape[0] == kwargs["number of surfaces"], "energies have incorrect shape"
    assert kwargs["frequencies"].shape[0] == kwargs["number of modes"], "frequencies have incorrect shape"

    log.debug("Saving model to {:s}".format(path_full))

    # convert each numpy array to a list
    for key, value in kwargs.items():
        if isinstance(value, (np.ndarray, np.generic)):
            kwargs[key] = value.tolist()

    # the old way involved passing in each parameter as nonkeyword argument
    # output_dictionary = {
    #     "number of modes":     N,
    #     "number of surfaces":  A,
    #     "energies":            kwargs["energies"].tolist(),
    #     "frequencies":         kwargs["frequencies"].tolist(),
    #     "linear couplings":    kwargs["linear_couplings"].tolist(),
    #     "quadratic couplings": kwargs["quadratic_couplings"].tolist(),
    # }

    with open(path_full, mode='w', encoding='UTF8') as target_file:
        target_file.write(json.dumps(kwargs))

    return
# ------------------------------------------------------------------------


def root_directories_exists(path_root, id_data, id_rho=0):
    path_data = (path_root + dir_vib).format(id_data)
    dir_list = [path_data]
    dir_list = [path_data + "electronic_structure/"]
    dir_list.extend([path_data + x for x in list_sub_dirs])
    dir_list.extend([path_data + dir_rho.format(id_rho) + x for x in list_sub_dirs])
    if False in map(os.path.isdir, dir_list):
        return False
    return True


def make_root_directories(path_root, id_data, id_rho=0):
    dir_list = [path_root]
    dir_list = [path_data + "electronic_structure/"]
    dir_list.extend([path_root + x for x in list_sub_dirs])
    dir_list.extend([path_root + dir_rho.format(id_rho) + x for x in list_sub_dirs])
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)
    return


def make_rho_directories(path_root, id_rho):
    dir_list = [path_root + dir_rho.format(id_rho) + x for x in list_sub_dirs]
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)
    return


# TODO - this currently does not support loading models with more than quadratic coupling
def load_model_from_JSON(path_full, energies=None,frequencies=None, linear_couplings=None, quadratic_couplings=None):
    """if only a path to the file is provided, arrays
    are returned othwerise provided arrays are filled with appropriate values"""
    log.debug("Loading model {:s}".format(path_full))
    # open the JSON file
    target_file = open(path_full, mode='r', encoding='UTF8')
    input_dictionary = json.loads(target_file.read())
    target_file.close()
    nmode = int(input_dictionary["number of modes"])
    nel = int(input_dictionary["number of surfaces"])

    # no arrays were provided so return newly created
    # arrays after filling them with the approriate values
    if frequencies is None:
        energies = np.empty((nel, nel), dtype=F64)
        energies[:] = np.array(input_dictionary["energies"], dtype=F64)

        frequencies = np.empty((nmode,), dtype=F64)
        frequencies[:] = np.array(input_dictionary["frequencies"], dtype=F64)

        linear_couplings = np.empty((nmode, nel, nel), dtype=F64)
        linear_couplings[:] = np.array(input_dictionary["linear couplings"], dtype=F64)

        quadratic_couplings = np.empty((nmode, nmode, nel, nel), dtype=F64)
        quadratic_couplings[:] = np.array(input_dictionary["quadratic couplings"], dtype=F64)

        return energies, frequencies, linear_couplings, quadratic_couplings

    # arrays were provided so fill them with the appropriate values
    else:
        assert(energies.shape == (nel, nel))
        assert(frequencies.shape == (nmode,))
        assert(linear_couplings.shape == (nmode, nel, nel))
        assert(quadratic_couplings.shape == (nmode, nmode, nel, nel))

        energies[:] = np.array(input_dictionary["energies"], dtype=F64)
        frequencies[:] = np.array(input_dictionary["frequencies"], dtype=F64)
        linear_couplings[:] = np.array(input_dictionary["linear couplings"], dtype=F64)
        quadratic_couplings[:] = np.array(input_dictionary["quadratic couplings"], dtype=F64)

    return


# TODO - this currently does not support loading models with more than quadratic coupling
def load_sample_from_JSON(path_full, energies=None, frequencies=None, linear_couplings=None, quadratic_couplings=None):
    """x"""

    log.debug("Loading rho model {:s}".format(path_full))
    # open the JSON file
    with open(path_full, mode='r', encoding='UTF8') as target_file:
        input_dictionary = json.loads(target_file.read())
    target_file = open(path_full, mode='r', encoding='UTF8')
    input_dictionary = json.loads(target_file.read())
    target_file.close()
    nmode = int(input_dictionary["number of modes"])
    nel = int(input_dictionary["number of surfaces"])

    # no arrays were provided so return newly created
    # arrays after filling them with the approriate values
    if frequencies is None:
        energies = np.empty((nel), dtype=F64)
        energies[:] = np.diag(np.array(input_dictionary["energies"], dtype=F64))

        frequencies = np.empty((nmode), dtype=F64)
        frequencies[:] = np.array(input_dictionary["frequencies"], dtype=F64)

        linear_couplings = np.empty((nmode, nel), dtype=F64)
        linear_couplings[:] = np.diagonal(np.array(input_dictionary["linear couplings"], dtype=F64), axis1=1, axis2=2)

        quadratic_couplings = np.empty((nmode, nmode, nel), dtype=F64)
        quadratic_couplings[:] = np.diagonal(np.array(input_dictionary["quadratic couplings"], dtype=F64), axis1=2, axis2=3)

        return energies, frequencies, linear_couplings, quadratic_couplings

    # arrays were provided so fill them with the appropriate values
    else:
        # HACK
        # print("HACKY BS - ")

        # ===================== DEPRECIATED =================================== #
        # ==== this code is only needed if the input sample model is still in the old format
        # ==== it will be eventually removed
        # assert(energies.shape == (nel,))
        # assert(frequencies.shape == (nmode,))
        # assert(linear_couplings.shape == (nmode, nel))
        # assert(quadratic_couplings.shape == (nmode, nmode, nel))
        # energies[:] = np.diag(np.array(input_dictionary["energies"], dtype=F64))
        # frequencies[:] = np.array(input_dictionary["frequencies"], dtype=F64)
        # linear_couplings[:] = np.diagonal(np.array(input_dictionary["linear couplings"], dtype=F64), axis1=1, axis2=2)
        # quadratic_couplings[:] = np.diagonal(np.array(input_dictionary["quadratic couplings"], dtype=F64), axis1=2, axis2=3)
        # ===================== DEPRECIATED =================================== #

        energies[:] = np.array(input_dictionary["energies"], dtype=F64)
        frequencies[:] = np.array(input_dictionary["frequencies"], dtype=F64)
        linear_couplings[:] = np.array(input_dictionary["linear couplings"], dtype=F64)
        quadratic_couplings[:] = np.array(input_dictionary["quadratic couplings"], dtype=F64)
    return


# this should not be used for the moment
def setup_input_params(directory_path, new_directory=False):
    """collate the input parameters and write to a file in the execution directory"""
    print("Preparing the {} directory\nIt is {} that I am a new directory".format(directory_path, new_directory))

    # some default file paths
    path_params = directory_path + "parameters/"
    path_results = directory_path + "results/"
    path_output = directory_path + "execution_output/"
    path_plots = directory_path + "plots/"
    path_rho_params = directory_path + "rho_0/" + "parameters/"
    path_rho_results = directory_path + "rho_0/" + "results/"
    path_rho_output = directory_path + "rho_0/" + "execution_output/"
    path_rho_plots = directory_path + "rho_0/" + "plots/"
    json_filename = "coupled_model.json"

    # where we store the input parameters
    division_quadratic_term = None
    division_linear_term = None
    frequency_range = None
    number_of_surfaces = None
    number_of_modes = None
    temperature_list = None
    sample_list = None
    bead_list = None

    def create_dirs(dir_path):
        os.makedirs(path_params,  exist_ok=True)
        os.makedirs(path_results, exist_ok=True)
        os.makedirs(path_output,  exist_ok=True)
        os.makedirs(path_plots,   exist_ok=True)
        os.makedirs(path_rho_params,  exist_ok=True)
        os.makedirs(path_rho_results, exist_ok=True)
        os.makedirs(path_rho_output,  exist_ok=True)
        os.makedirs(path_rho_plots,   exist_ok=True)
        return

    def cleanup_dirs(dir_path):
        os.system("rm -r {:}*".format(path_results))
        os.system("rm -r {:}*".format(path_output))
        os.system("rm -r {:}*".format(path_plots))
        return

    if new_directory:
        create_dirs(directory_path)
    else:
        print("Do you want to remove all data in {:}? (Y/N) ".format(directory_path))
        choice = input().lower()
        if choice in set(["yes", "y", "ye"]):
            cleanup_dirs(directory_path)
            print("Cleaned up {:}".format(directory_path))
        else:
            print("Proceedeing to only overwrite the JSON parameter file\n")

    # copy the model_parameters_source.txt file to the new directory
    # shutil.copy(filename_source, path_params + filename_source[2:])

    generating_test_data = False

    if(generating_test_data):
        energies, frequencies, linear_terms, quadratic_terms = generate_vibronic_model_data(source_file="./model_parameters_source.txt", nonadiabatic=True)
    else:
        # nsurf, nmode, energies, frequencies, linear_terms, quadratic_terms = read_model_auto_file("./vibron_data/cp.auto")
        path = '/home/ngraymon/chem740/work_dir/scripting/ammonia/ammonia_vibron'
        nsurf, nmode, energies, frequencies, linear_terms, quadratic_terms = read_model_h_file(path)

    kwargs = {"number of modes": frequencies.shape[0],
              "number of surfaces": energies.shape[0],
              "energies": energies,
              "frequencies": frequencies,
              "linear couplings": linear_terms,
              "quadratic couplings": quadratic_terms,
              }
    save_model_to_JSON(path_params+json_filename, **kwargs)


def get_nmode_nsurf_from_coupled_model(id_model):
    """find nmodes and nsurfs for coupling_model.json files"""
    # checkOS()
    directory_path = (path_default_root + dir_vib + "parameters/").format(id_model)

    json_filename = "coupled_model.json"
    path_full = directory_path + json_filename

    with open(path_full, mode='r', encoding='UTF8') as target_file:
        input_dictionary = json.loads(target_file.read())

    number_of_modes = input_dictionary["number of modes"]
    number_of_surfaces = input_dictionary["number of surfaces"]
    return number_of_modes, number_of_surfaces


def get_nmode_nsurf_from_coupled_modelff(path_full):
    """find nmodes and nsurfs for sampling_model.json files by using a filepath"""
    with open(path_full, mode='r', encoding='UTF8') as target_file:
        input_dictionary = json.loads(target_file.read())

    number_of_modes = input_dictionary["number of modes"]
    number_of_surfaces = input_dictionary["number of surfaces"]
    return number_of_modes, number_of_surfaces


def get_nmode_nsurf_from_sampling_model(id_model, id_rho):
    """find nmodes and nsurfs for sampling_model.json files"""
    # checkOS()
    directory_path = (path_default_root + dir_vib + dir_rho + "parameters/").format(id_model, id_rho)

    json_filename = "sampling_model.json"
    path_full = directory_path + json_filename

    with open(path_full, mode='r', encoding='UTF8') as target_file:
        input_dictionary = json.loads(target_file.read())

    number_of_modes = input_dictionary["number of modes"]
    number_of_surfaces = input_dictionary["number of surfaces"]
    return number_of_modes, number_of_surfaces


def get_nmode_nsurf_from_sampling_modelff(path_full):
    """find nmodes and nsurfs for sampling_model.json files by using a filepath"""
    with open(path_full, mode='r', encoding='UTF8') as target_file:
        input_dictionary = json.loads(target_file.read())

    number_of_modes = input_dictionary["number of modes"]
    number_of_surfaces = input_dictionary["number of surfaces"]
    return number_of_modes, number_of_surfaces


def remove_coupling_from_model(path_source, path_destination):
    """reads in a model and sets all the coupling parameters to zero"""
    checkOS()

    e, w, l, q = load_model_from_JSON(path_source)
    numModes = w.shape[0]
    numStates = e.shape[0]
    Modes = range(numModes)
    States = range(numStates)

    energy = np.diagonal(e).copy()
    linear = np.diagonal(l, axis1=1, axis2=2).copy()
    quadratic = np.diagonal(q, axis1=2, axis2=3).copy()
    kwargs = {"number of modes": numModes,
              "number of surfaces": numStates,
              "energies": energy,
              "frequencies": w,
              "linear couplings": linear,
              "quadratic couplings": quadratic,
              }
    save_model_to_JSON(path_destination, **kwargs)
    return


def create_harmonic_model(FS):
    """wrapper function to refresh harmonic model"""
    checkOS()
    source = FS.path_vib_params + "coupled_model.json"
    dest = FS.path_vib_params + "harmonic_model.json"
    # source = (FS.path_default_root + dir_vib + "parameters/coupled_model.json").format(id_model)
    # dest = (FS.path_default_root + dir_vib + "parameters/harmonic_model.json").format(id_model)
    remove_coupling_from_model(source, dest)
    s = "Created harmonic model {:s} by removing coupling from {:s}"
    log.debug(s.format(dest, source))
    return dest


def create_basic_sampling_model(FS):
    """wrapper function to make the simplest sampling model"""
    checkOS()
    source = create_harmonic_model(FS)
    dest = FS.path_rho_params + "sampling_model.json"

    if os.path.isfile(dest):
        s = "Sampling model {:s} already exists!"
        log.debug(s.format(dest))

    shutil.copyfile(source, dest)

    s = "Created sampling model {:s} by copying {:s}"
    log.debug(s.format(dest, source))
    return dest


# this should not be used for the moment
def create_new_execution_directory(path_root=None, id_data=None):
    """create or overwrite an execution directory"""
    # checkOS()

    # data directories are named "data_set_XXX" where XXX can be an integer
    # from 1 to DIR_MAX
    DIR_MAX = int(1e6)
    DIR_RANGE = range(1, DIR_MAX)

    # DIR_ARRAY = np.core.defchararray.add(np.full(shape=DIR_MAX, fill_value="data_set_", dtype=string_), np.arange(start=1, stop=DIR_MAX), dtype=string_)
    # print(DIR_ARRAY)

    if path_root is None and id_data is None:
        counter = it.count(1)
        dir_base = path_default_root + dir_vib
        for directory_exists in map(os.path.isdir, map(dir_base.format, counter)):
            if not directory_exists:
                free_dir_id = next(counter)-1
                new_directory = dir_base.format(free_dir_id)
                s = "It appears {:} is available, writing model ..."
                print(s.format(new_directory))
                setup_input_params(new_directory, new_directory=True)
                break
        else:
            raise Exception("Okay... we had issues, you used up the 1e6 possible "
                            "data set names?????"
                            )

    elif path_root is None:
        full_path = (path_default_root + dir_vib).format(id_data)

        if os.path.isdir(full_path):
            print("We discovered a previous execution at {:}, "
                  "do you wish to proceed? (Y/N)".format(full_path)
                  )
            choice = input().lower()
            if choice in set(["yes", "y", "ye"]):
                setup_input_params(full_path)
                print("Successfully created new model parameters at:{:}".format(full_path))
                return
            print("Stopping\n")
            return

        setup_input_params(full_path, new_directory=True)
        print("Successfully created model params at:\n{:}".format(full_path))
        return

    else:
        full_path = (path_root + dir_vib).format(id_data)
        if os.path.isdir(full_path):
            print("We discovered a previous execution at {:}, "
                  "do you wish to proceed? (Y/N)".format(full_path)
                  )
            choice = input().lower()
            if choice in set(["yes", "y", "ye"]):
                setup_input_params(full_path)
                print("Successfully created new model parameters at:{:}".format(full_path))
                return
            print("Stopping\n")
            return

        setup_input_params(full_path, new_directory=True)
        print("Successfully created model params at:\n{:}".format(full_path))
        return


def verify_execution_directory_exists(id_data, path_root=None):
    """ if directory structure exists does nothing, creates the file structure otherwise"""
    # checkOS()

    # assume default
    if path_root is None:
        path_root = path_default_root

    files = file_structure.FileStructure(path_root, id_data)
    if files.directories_exist():
        return

    files.make_directories()
    return


if (__name__ == "__main__"):
    if len(sys.argv) == 3:
        path_root = sys.argv[1]
        assert(sys.argv[2].isnumeric() and int(sys.argv[2]) >= 1)
        id_data = int(sys.argv[2])
        create_new_execution_directory(path_root=path_root, id_data=id_data)
    elif len(sys.argv) == 2:
        assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)
        id_data = int(sys.argv[1])
        create_new_execution_directory(id_data=id_data)
    else:
        create_new_execution_directory()

    # read_model_h_file('/home/ngraymon/chem740/work_dir/scripting/ammonia/ammonia_vibron')
