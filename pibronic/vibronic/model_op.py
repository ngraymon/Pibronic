""" module that handles parsing model.op files"""

# system imports
import itertools as it
import mmap

# third party imports
import numpy as np
import parse

# local imports
# from ..log_conf import log
# from .. import constants
# from ..data import file_structure
# from ..data import file_name
from .. import helper
from .vibronic_model_keys import VibronicModelKeys as VMK


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

    # TODO - should remove the use of globals
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

    # TODO - should remove the use of globals
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

    return


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

    # TODO - should remove the use of globals
    global numStates, numModes, States, Modes, size
    # we will overwrite these default values
    numStates = numModes = States = Modes = size = 0

    helper.verify_file_exists(path_file_op)

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

    # TODO - do we not account for the symmetry in surfaces? this might be incorrect?

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
