""" functions which prepare the environment for executing matplotlib on the server """

# system imports
import subprocess

# third party imports
import matplotlib as mpl
import matplotlib.pyplot as plt

# local imports


def prepare_mpl_rc_file(pretty_but_slow=False):
    """ TODO - this needs to be refactored and cleaned up (it is sufficiently functional for the moment) """

    # TODO - this doesn't seem to work?
    # mpl.rcParams['backend'] = "agg"  # For the server we need to force the use of Agg
    # mpl.rcParams['backend'] = "ps"  # We need to use the postscript backend to generate eps files

    plt.switch_backend("agg")

    if pretty_but_slow:
        # change the font
        mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
        # mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Sans serif']})
        # mpl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    mpl.rc('text', usetex=True)  # using LaTeX
    # we need to load the amsmath package to use the \text{} command
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    return


def load_latex_module_on_server(version="2017"):
    """ load the texlive module so that we can make plots with latex
    this function will only work on our local server
    TODO - there should be a replacement for local execution and execution on other servers"""
    cmd = ['/usr/bin/modulecmd', 'python', 'load', f'texlive/{version:s}']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, error = p.communicate()
    exec(out)  # this is necessary!
    return
