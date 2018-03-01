# Pibronic
=======================

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

----
