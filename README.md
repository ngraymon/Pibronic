# Pibronic
[![PyPI](https://img.shields.io/pypi/v/pibronic.svg)](https://pypi.org/project/pibronic/)
[![Travis](https://img.shields.io/travis/ngraymon/Pibronic.svg)](https://travis-ci.org/ngraymon/Pibronic)
[![Coverage Status](https://codecov.io/gh/ngraymon/Pibronic/branch/master/graph/badge.svg)](https://codecov.io/gh/ngraymon/Pibronic)
----

A python package that handles the creation of vibronic models using ACESII and VIBRON
as well as the calculation of thermodynamic properties using those vibronic models.

The package provides an implementation of a PIMC method that calculates thermodynamic properties of quantum mechanical systems described by vibronic models (Hamiltonians in the diabatic representation).

It includes  scripts to generate vibronic models using the computational packages ACESII and VIBRON.

Additional scripts are provided for:
- submitting and managing jobs in a server environment running SLURM.
- collating and processing output from the PIMC calculations
- plotting the processed data


----

Currently undergoing refactoring, specifically file I/O and fileStructure
Ancillary code will be uploading in the near future
The main PIMC simulation code is provided in minimal.py

In Progress:
- link postprocessing to jackknife
- increase coverage of tests

To do list:
- refresh jackknife
- inject file_name and file_structure into older modules
- prune vibronic_model_io
- integrate FileStructure with electronic_structure
- third pass on posprocessing
- clean up server job submission scripts


Long term goals:
- clean up jackknife
- provide model systems
- generate test date for fixed rng seeds
- travis.ci + appveyor + shields.io?

----
