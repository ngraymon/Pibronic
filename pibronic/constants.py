# really simple module to store the choice of boltzman constant
# between all files
#
#
#
import numpy as np

# reference http://ws680.nist.gov/publication/get_pdf.cfm?pub_id=920687
# reference http://ws680.nist.gov/publication/get_pdf.cfm?pub_id=920686

# joules / ev
nist_j_per_ev = np.float64(1.6021766208e-19)

# 1 / mol
avogadro = np.float64(6.022140857e+23)

# joules / kelvin
# old_boltzman =np.float64(8.6173303e-5)

# joules / kelvin
# boltzman = np.float64(1.38064852e-23)

# ev / kelvin
boltzman = np.float64(1.38064852e-23) / nist_j_per_ev

hbar = 1.0

delta_beta = 2.0e-4

# convert wavenumbers to electronVolts and back
wavenumber_per_eV = 8065.6


# the number of bytes in a giga-byte
GB_per_byte = 1e9

maximum_memory_per_node = 128  # in giga-bytes


def beta(temperature):
    """returns a value of beta for a given temperature in Kelvin"""
    val = 1. / (temperature * boltzman)
    return val
