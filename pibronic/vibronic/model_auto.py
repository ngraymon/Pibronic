""" module that handles parsing model.auto files"""

# system imports
# import itertools as it
# import mmap

# third party imports
import fortranformat as ff  # Fortran format for VIBRON
import numpy as np
from numpy import float64 as F64

# local imports
# from ..log_conf import log
# from .. import constants
# from ..data import file_structure
# from ..data import file_name
# from .. import helper
# from .vibronic_model_keys import VibronicModelKeys as VMK

""" pre-defined header formats """
title_header = ff.FortranRecordReader('(A50, A20)')
coefficient_header = ff.FortranRecordReader('2(F12.6)')
gradient_header = ff.FortranRecordReader('(t3, A15, i4, A15, F10.2)')


def read_header(f_iter, nel, nmode, frequencies):
    numgrid = ff.FortranRecordReader('(A26, i4)').read(next(f_iter))[1]  # read the type of grid in ACESII
    num_irreps = ff.FortranRecordReader('(A36, i4)').read(next(f_iter))[1]  # number of vibrational irreps
    nmodes = ff.FortranRecordReader('(A30, i4)').read(next(f_iter))[1]  # read the total number of modes per symmetry

    # we can either grab the frequencies from the ref_freq_data, or the parent hessian
    if(False):
        # the frequency given here is in cm-1
        frequencies[:] = np.asarray(ff.FortranRecordReader('(A31, 12F8.2)').read(next(f_iter))[1:], dtype=F64)
    else:
        next(f_iter)

    return


def fill_in_excitation_energies(f_iter, el_range, excitation_energies):
    """ does what it says """
    for a in el_range:
        excitation_energies[a, :] = coefficient_header.read(next(f_iter))


def fill_in_linear_couplings(f_iter, el_range, mode_range, linear_couplings):
    """ does what it says """
    for i in mode_range:
        gradient_header.read(next(f_iter))  # should 'read in' the header
        for a in el_range:
            linear_couplings[i, a, :] = coefficient_header.read(next(f_iter))
    return


def fill_in_diagonal_quadratic_couplings(f_iter, el_range, mode_range, quadratic_couplings):
    """ does what it says """
    for i in mode_range:
        gradient_header.read(next(f_iter))
        for a in el_range:
            quadratic_couplings[i, i, a, :] = coefficient_header.read(next(f_iter))
    return


def fill_in_offdiagonal_quadratic_couplings(f_iter, el_range, mode_range, quadratic_couplings):
    """ does what it says """
    for i in mode_range:
        for j in range(i):
            next(f_iter)  # 'reading in' the first line/title
            for a in el_range:
                quadratic_couplings[i, j, a, :] = coefficient_header.read(next(f_iter))

    for a in el_range:
        for b in el_range:
            quadratic_couplings[:, :, a, b] += np.tril(quadratic_couplings[:, :, a, b], k=-1).T

    return


def confirm_symmetry_in_surfaces(el_range, linear_couplings, quadratic_couplings):
    """ check for symmetry in surfaces """
    for a in el_range:
        for b in el_range:
            assert(np.allclose(linear_couplings[:, a, b], linear_couplings[:, b, a]))
            assert(np.allclose(quadratic_couplings[:, :, a, b], quadratic_couplings[:, :, b, a]))
    return


def confirm_symmetry_in_modes(mode_range, quadratic_couplings):
    """ check for symmetry in modes """
    for i in mode_range:
        for j in mode_range:
            assert(np.allclose(quadratic_couplings[i, j, :, :], quadratic_couplings[j, i, :, :]))
    return


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
    el_range = range(nel)
    mode_range = range(nmode)

    # Read file (Iterator object)
    file_object = open(filename, 'r')

    # first lets strip out all the blank lines and lines like ------
    lines = [line.rstrip() for line in file_object.readlines() if ((len(line.strip()) != 0) and ("--------" not in line.strip()))]
    f_iter = iter(lines)  # make an iterator

    # =====
    # Title
    # =====

    title_header.read(next(f_iter))

    # ==========================
    # 0. Header and Normal modes
    # ==========================

    # Initialize Variables
    excitation_energies = np.zeros((nel, nel))
    frequencies = np.zeros((nmode))
    linear_couplings = np.zeros((nmode, nel, nel))
    quadratic_couplings = np.zeros((nmode, nmode, nel, nel))

    read_header(f_iter, nel, nmode, frequencies)

    # ==================
    # 1. Parent Gradient
    # ==================

    title_header.read(next(f_iter))
    # gradient_header_parent = ff.FortranRecordReader('(t3, A17, i4, t22, A15, E16.10)')
    for i in mode_range:
        next(f_iter)

    # =================
    # 2. Parent Hessian
    # =================

    title_header.read(next(f_iter))
    # hessian_header_parent = ff.FortranRecordReader('(t3, A17, i4, t22, A15, F8.2, A5, F12.6, A5)')
    try:
        for i in mode_range:
            # store the value incase exception occurs
            temp_str = next(f_iter)
            frequencies[i] = float(temp_str.partition("cm-1")[2].partition("eV")[0])
    except ValueError as err_obj:
        print("Having issues parsing the Hessian parent energy, why are the frequences not digits?", temp_str, temp_str.partition("cm-1")[2], temp_str.partition("cm-1")[2].partition("eV")[0])
        raise err_obj

    # ========================
    # 3. Reference Hamiltonian
    # ========================

    title_header.read(next(f_iter))
    for a in el_range:
        excitation_energies[a, :] = coefficient_header.read(next(f_iter))
    fill_in_excitation_energies(f_iter, el_range, excitation_energies)

    # ===============================
    # 4. Linear couplings (gradients)
    # ===============================

    title_header.read(next(f_iter))
    fill_in_linear_couplings(f_iter, el_range, mode_range, linear_couplings)

    # ========================================================
    # 5. Quadratic couplings [Diagonal corrections of Hessian]
    # ========================================================

    title_header.read(next(f_iter))
    fill_in_diagonal_quadratic_couplings(f_iter, el_range, mode_range, quadratic_couplings)

    # ============================================================
    # 6. Quadratic couplings [Off-diagonal corrections of Hessian]
    # ============================================================

    title_header.read(next(f_iter))
    fill_in_offdiagonal_quadratic_couplings(f_iter, el_range, mode_range, quadratic_couplings)

    confirm_symmetry_in_surfaces(el_range, linear_couplings, quadratic_couplings)
    confirm_symmetry_in_modes(mode_range, quadratic_couplings)

    return nel, nmode, excitation_energies, frequencies, linear_couplings, quadratic_couplings
