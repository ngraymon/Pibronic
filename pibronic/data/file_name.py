"""
Bookkeeping provided by file_name module

Store all naming conventions in one place
Keeps file naming consistant over the many modules if changes need to be made
"""

# system imports
# third party imports
# local imports

# TODO - conside the idea of making the template strings parameterizable as well as including different generating functions - seems like a lot of overhead for little gain?


""" the execution output of a pimc run
D - refers to the id of the data set
R - refers to the id of the sampling set (rho)
P - refers the number of beads (int)
T - refers to the tempearture in Kelvin (float-2 places)
 - number of samples used to obtain results inside file (int)
"""
_execution_output = "D{:s}_R{:s}_P{:s}_T{:s}.o{:s}"


def execution_output(D="{D:d}", R="{R:d}", P="{P:d}", T="{T:.2f}", J="{J:d}"):
    """ wrapper function to allow parameterized search for files
        can pass in any regular expression
        most commonly will just pass in "*" to search for all files of a given argument
    """
    return _execution_output.format(D, R, P, T, J)


""" the results of a pimc run
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


""" R values used for training ML algorithms
P - refers the number of beads (int)
T - refers to the tempearture in Kelvin (float-2 places)
J - jobnumber, default is zero (int)
"""
_training_data_input = "P{:s}_T{:s}_J{:s}_training_data_input.npz"


def training_data_input(P="{P:d}", T="{T:.2f}", J="{J:d}"):
    """ wrapper function to allow parameterized search for files
        can pass in any regular expression
        most commonly will just pass in "*" to search for all files of a given argument
    """
    return _training_data_input.format(P, T, J)


""" output of a g(R) run - used for training ML algorithms
P - refers the number of beads (int)
T - refers to the tempearture in Kelvin (float-2 places)
J - jobnumber, default is zero (int)
"""
_training_data_g_output = "P{:s}_T{:s}_J{:s}_training_data_g_output.npz"


def training_data_g_output(P="{P:d}", T="{T:.2f}", J="{J:d}"):
    """ wrapper function to allow parameterized search for files
        can pass in any regular expression
        most commonly will just pass in "*" to search for all files of a given argument
    """
    return _training_data_g_output.format(P, T, J)


""" output of a rho(R) run - used for training ML algorithms
P - refers the number of beads (int)
T - refers to the tempearture in Kelvin (float-2 places)
J - jobnumber, default is zero (int)
"""
_training_data_rho_output = "P{:s}_T{:s}_J{:s}_training_data_rho_output.npz"


def training_data_rho_output(P="{P:d}", T="{T:.2f}", J="{J:d}"):
    """ wrapper function to allow parameterized search for files
        can pass in any regular expression
        most commonly will just pass in "*" to search for all files of a given argument
    """
    return _training_data_rho_output.format(P, T, J)


# TODO - maybe only need one file name for both rho and sos???
""" output of SOS for model diagonal in electronic states
B - refers the number of basis functions to obtain results inside file (int)
"""

""" contains the parameters which describe the systems Hamiltonian
this is the full model
"""
coupled_model = "coupled_model.json"

""" contains the original parameters (diagonal) which describe the systems Hamiltonian
before we did a unitary transformation
"""
original_model = "original_coupled_model.json"

""" contains the parameters which describe the harmonic operator
this is all the Kinetic terms + any diagonal terms that can be described by a Harmonic oscillator
this can be used as a simplistic sampling distribution rho_0
"""
harmonic_model = "harmonic_model.json"

""" contains the parameters which describe a sampling model
this can be anything which is diagonal in the electronic states
"""
sampling_model = "sampling_model.json"

""" contains any parameters which can be obtained through analytical methods
this commonly stores properties of the sampling model at different temperatures
"""
analytic_results = "analytic_results.json"

# TODO - maybe include the directories and so forth inside file_name? or maybe in file_structure?
# list_sub_dirs = [
#     "parameters/",
#     "results/",
#     "execution_output/",
#     "plots/",
# ]
