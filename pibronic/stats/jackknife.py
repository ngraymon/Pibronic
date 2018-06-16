# jackknife.py - although might want to rename this to statistical_analysis.py?

# system imports


# local imports
from .. import constants
from ..constants import boltzman

# third party imports
import numpy as np


def calculate_alpha_jackknife_terms(*args):
    """x"""
    # LN is ratio_plus == (g_plus / rho)
    # RN is ratio_minus == (g_minus / rho)
    X, delta_beta, ratio, LN, alpha_plus, RN, alpha_minus = args

    # first start with the full sum
    jk_ratio = np.full(shape=X, fill_value=np.sum(ratio))

    # only this needs special handling
    jk_LN = np.full(shape=X, fill_value=np.sum(LN))
    # jk_LD = np.full(shape=X, fill_value=np.sum(LD))
    jk_RN = np.full(shape=X, fill_value=np.sum(RN))
    # jk_RD = np.full(shape=X, fill_value=np.sum(RD))

    # now subtract each term from the sum
    jk_ratio -= ratio
    jk_LN -= LN
    # jk_LD -= LD
    jk_RN -= RN
    # jk_RD -= RD

    # pre normalize
    jk_ratio /= (X - 1)
    jk_LN /= (X - 1)
    # jk_LD /= (X - 1)
    jk_RN /= (X - 1)
    # jk_RD /= (X - 1)

    # now build the first and second symmetric derivative
    jk_sym_d1 = jk_LN * alpha_plus
    jk_sym_d1 -= jk_RN * alpha_minus
    jk_sym_d1 /= (2. * delta_beta)  # constant factor
    #
    jk_sym_d2 = jk_LN * alpha_plus
    jk_sym_d2 -= 2. * jk_ratio
    jk_sym_d2 += jk_RN * alpha_minus
    jk_sym_d2 /= pow(delta_beta, 2)  # constant factor

    ret = [jk_ratio,
           jk_sym_d1,
           jk_sym_d2]

    return ret


def calculate_jackknife_term(length, array):
    """calculate the jackknife term for a given array"""

    # start with the full sum
    jk_term = np.full(shape=length, fill_value=np.sum(array))
    # subtract each term from the sum
    jk_term -= array
    # normalize
    jk_term /= length

    return jk_term


def calculate_jackknife_terms(length, list_of_arrays):
    return [calculate_jackknife_term(length, array) for array in list_of_arrays]


def estimate_jackknife(*args):
    """"""
    X, T, delta_beta, input_dict, jk_g_rho, jk_sym1, jk_sym2 = args

    # # Energy - jackknife edition
    # the jackknife estimator
    f_EJ = -1. * jk_sym1 / jk_g_rho
    #
    E = X * input_dict["E"]
    E -= (X - 1.) * np.mean(f_EJ)
    # error bars
    E_err = np.sqrt(X - 1.) * np.std(f_EJ, ddof=1)

    # # Heat Capacity - jackknife edition
    # the jackknife estimator
    f_CJ = jk_sym2 / jk_g_rho
    f_CJ -= pow(f_EJ, 2.)
    f_CJ /= boltzman * pow(T, 2.)
    #
    Cv = X * input_dict["Cv"]
    Cv -= (X - 1.) * np.mean(f_CJ)
    # error bars
    Cv_err = np.sqrt(X - 1.) * np.std(f_CJ, ddof=1)

    # easy to access storage
    return_dictionary = {"E": E,   "E error": E_err,
                         "Cv": Cv, "Cv error": Cv_err,
                         }
    return return_dictionary


if (__name__ == "__main__"):
    pass
