""" module that handles parsing model.h files"""

# system imports
import itertools as it
import mmap

# third party imports
import numpy as np

# local imports
from .. import helper
from ..log_conf import log
from .vibronic_model_keys import VibronicModelKeys as VMK


def get_number_of_electronic_states(path, file):
    """ return the number of electronic states (int) taken from file """
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
        raise Exception(s.format(path, targetString))

    return number_of_electronic_states


def get_number_of_normal_modes(path, file):
    """ return the total number of normal modes (int) and symmetric modes (int) taken from file """
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
        raise Exception(s.format(path, targetString))

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
        raise Exception(s.format(path, targetString))

    return number_of_normal_modes, number_of_symmetric_modes


def extract_normal_mode_frequencies(path, file, frequency_array, symmetric_modes):
    """fill the array frequency_array with appropriate values from the memmap'ed file"""
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


def extract_energies(path, memmap, energies):
    """fill the array energies with appropriate values from the memmap'ed file"""
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
    return


def extract_linear_couplings(path, memmap, coupling_terms, frequencies):
    """fill the array coupling_terms with appropriate values from the memmap'ed file
    the frequencies need to be provided in wavenumbers"""
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
    return


def extract_quadratic_couplings(path, memmap, coupling_terms, frequencies):
    """fill the array coupling_terms with appropriate values from the memmap'ed file
    the frequencies need to be provided in wavenumbers"""
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
    return


def extract_offdiag_quadratic_couplings(path, memmap, coupling_terms, frequencies):
    """fill the array coupling_terms with appropriate values from the memmap'ed file
    the frequencies need to be provided in wavenumbers"""
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
                raise Exception(s.format(w1, path, beginString))

            # find the next block of quadratic coupling terms
            next_block = memmap.find(str(w2).encode(encoding="utf-8"), memmap.tell(), end)
            # error handling
            if next_block == -1:
                s = ("Frequency {:f} in wavenumber_freq did not "
                     "match any frequencies in file {:s} "
                     "while parsing the {:s} region."
                     )
                raise Exception(s.format(w2, path, beginString))

            # go there
            memmap.seek(next_block)
            # skip header
            memmap.readline()

            # store each line in the array
            for a in States:
                line = memmap.readline().decode(encoding="utf-8")
                coupling_terms[idx1, idx2, a, :] = line.split()
    return


def extract_cubic_couplings(path, memmap, coupling_terms, frequencies):
    """ not implemented at this time """
    return


def extract_quartic_couplings(path, memmap, coupling_terms, frequencies):
    """ not implemented at this time """
    return


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

    # store the energy offsets, and all the coupling terms
    with open(path_file_h, "r+b") as source_file:
        # access the file using memory map for efficiency
        with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as mm:
            extract_energies(path_file_h, mm, excitation_energies, frequencies)
            extract_linear_couplings(path_file_h, mm, wavenumber_freq, linear_couplings, frequencies)
            # extract_quadratic_couplings(path_file_h, mm, wavenumber_freq, quadratic_couplings, frequencies)
            # extract_offdiag_quadratic_couplings(path_file_h, mm, wavenumber_freq, quadratic_couplings, frequencies)
            # extract_cubic_couplings(path_file_h, mm, wavenumber_freq, cubic_couplings, frequencies)
            # extract_quartic_couplings(path_file_h, mm, wavenumber_freq, quartic_couplings, frequencies)
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
