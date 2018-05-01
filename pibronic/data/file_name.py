"""
Bookkeeping provided by file_name module

Store all naming conventions in one place
Keeps file naming consistant over the many modules if changes need to be made
"""

# system imports
# third party imports
# local imports

# TODO - conside the idea of making the template strings parameterizable as well as including different generating functions - seems like a lot of overhead for little gain?

""" output of a pimc run
P - refers the number of beads (int)
T - refers to the tempearture in Kelvin (float-2 places)
J - jobnumber, default is zero (int)
"""
_pimc = "P{:s}_T{:s}_J{:s}_data_points.npz"


def pimc(P="{P:d}", T="{T:.2f}", J="{J:d}"):
    """ wrapper function to allow parameterized search for files
        can pass in any regular expression
        most commonly will just pass in "*" to search for all files of a given argument
    """
    return _pimc.format(P, T, J)


""" output of postprocess + jackknife
P - refers the number of beads (int)
T - refers to the tempearture in Kelvin (float-2 places)
X - number of samples used to obtain results inside file (int)
"""
_jackknife = "P{:s}_T{:s}_X{:s}_thermo"


def jackknife(P="{P:d}", T="{T:.2f}", X="{X:d}"):
    """ wrapper function to allow parameterized search for files
        can pass in any regular expression
        most commonly will just pass in "*" to search for all files of a given argument
    """
    return _jackknife.format(P, T, X)


""" output of SOS for coupled model
B - refers the number of basis functions to obtain results inside file (int)
"""
_sos = "sos_B{:s}.json"


def sos(B="{B:d}"):
    """ wrapper function to allow parameterized search for files
        can pass in any regular expression
        most commonly will just pass in "*" to search for all files of a given argument
    """
    return _sos.format(B)


# TODO - maybe only need one file name for both rho and sos???
""" output of SOS for model diagonal in electronic states
B - refers the number of basis functions to obtain results inside file (int)
"""

""" contains the parameters which describe the systems Hamiltonian
this is the full model
"""
coupled_model = "coupled_model.json"

""" contains the parameters which describe the harmonic operator
this is all the Kinetic terms + any diagonal terms that can be described by a Harmonic oscillator
this can be used as a simplistic sampling distribution rho_0
"""
harmonic_model = "harmonic_model.json"

""" contains the parameters which describe a sampling model
this can be anything which is diagonal in the electronic states
"""
sampling_model = "sampling_model.json"

# TODO - maybe include the directories and so forth inside file_name? or maybe in file_structure?
# list_sub_dirs = [
#     "parameters/",
#     "results/",
#     "execution_output/",
#     "plots/",
# ]
