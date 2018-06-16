# Pibronic
[![PyPI](https://img.shields.io/pypi/v/pibronic.svg)](https://pypi.org/project/pibronic/)
[![Travis](https://img.shields.io/travis/ngraymon/Pibronic.svg)](https://travis-ci.org/ngraymon/Pibronic)
[![Coverage Status](https://codecov.io/gh/ngraymon/Pibronic/branch/master/graph/badge.svg)](https://codecov.io/gh/ngraymon/Pibronic)
[![arXiv Link](https://img.shields.io/badge/arXiv%3A-1805.05971-blue.svg)](https://arxiv.org/abs/1805.05971)
[![DOI Link](https://img.shields.io/badge/DOI-10.1063%2F1.5025058-blue.svg)](https://aip.scitation.org/doi/10.1063/1.5025058)
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
- plotting code and analysis code
- increasing coverage of tests
- third pass on posprocessing
- inject file_name and file_structure into older modules

To do list:
- refresh jackknife
- refresh server code
- write basic tests of server code
- Create discrete numerical test case to verify jackknife output
- prune vibronic_model_io
- integrate FileStructure with electronic_structure
- clean up server job submission scripts


Long term goals:
- clean up jackknife
- provide model systems
- generate test date for fixed rng seeds
- appveyor?

----
Thanks
--------------------
This package wouldn't have been possible without the help of [Dmitri Iouchtchenko](https://github.com/0 "GitHub Page")
