"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
https://docs.python.org/3/distutils/setupscript.html#distutils-installing-scripts
"""
import subprocess
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


def julia_is_installed():
    """verify that the right version of Julia (the programming language) is installed"""
    result = subprocess.run(["julia", "--version"], universal_newlines=True, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.stdout.startswith("julia version"):
        bits = result.stdout.split(sep=".")
        # if the version is 1.X.X or 0.7.X is installed
        if int(bits[0][-1]) >= 1 or int(bits[1]) >= 7:
            print("It appears julia 1.X.X or 0.7.X is installed, install should proceed successfully")
            return True
        else:
            print("It appears julia is installed but the version is too old please update to at least 1.0.0 or 0.7.0")
    else:
        print("Julia is not installed and so VibronicToolkit cannot be installed",
              "some features such as: analytic/sos/trotter will not be available")
    return False


def install_VibronicToolkit():
    """attempts to install the Julia package VibronicToolkit (developed by Dmitri Iouchtchenko)
    we link to a fork of the package which has modified bin scripts designed to provide formatted output for Pibronic's julia_wrapper.py"""
    url = "https://github.com/ngraymon/VibronicToolkit.jl.git"
    package_name = "VibronicToolkit"
    branch_name = "integrated"

    # should add a try catch so that Julia doesn't error out if the package is already installed?
    # TODO - update this section once the standard approach is finalized and bugs are ironed out in the v1.0.X versions of Julia

    # This approach is functional but there is probably a better way to do this
    cmd = 'using Pkg;'
    cmd += f'Pkg.add(PackageSpec(url="{url:s}", rev="{branch_name:s}"));'
    subprocess.run(['julia', '-e', cmd])

    # we trust that the install was successful
    # although it would be nice to have a clean way to confirm the install was successful
    return


def install_VibronicToolkit_in_development_mode():
    """attempts to install the Julia package VibronicToolkit (developed by Dmitri Iouchtchenko)
    in development mode
    we link to a fork of the package which has modified bin scripts designed to provide formatted output for Pibronic's julia_wrapper.py"""
    url = "https://github.com/ngraymon/VibronicToolkit.jl.git"
    package_name = "VibronicToolkit"
    branch_name = "integrated"

    # should add a try catch so that Julia doesn't error out if the package is already installed?
    # TODO - update this section once the standard approach is finalized and bugs are ironed out in the v1.0.X versions of Julia

    # This approach is functional but there is probably a better way to do this
    cmd = 'using Pkg;'
    cmd += f'Pkg.develop(PackageSpec(url="{url:s}", rev="{branch_name:s}"));'
    subprocess.run(['julia', '-e', cmd])

    # we trust that the install was successful
    # although it would be nice to have a clean way to confirm the install was successful
    return


def read_md(filename):
    """use Pandoc magic to convert Markdown to RST for uploading to PyPi"""
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, filename), encoding='utf-8') as f:
        try:
            from pypandoc import convert_text
            return convert_text(f.read(), 'rst', format="md")
        except ImportError:
            print("warning: pypandoc module not found, could not convert Markdown to RST")
            return f.read()


VERSION = (
            0,
            3,
            0,
            'dev',
            0
            )


def get_version(version=None):
    """Return version (X.Y[.Z]) from VERSION."""
    parts = 2 if version[2] == 0 else 3
    return '.'.join(str(x) for x in version[:parts])


version = get_version(VERSION)

REQUIRED_PYTHON = (3, 5)

EXCLUDE_FROM_PACKAGES = [
                         'contrib',
                         'docs',
                         'tests',
                         'dev',
                        ]


def setup_package():
    """ wapper for setup() so that we can check if julia is installed first """

    if julia_is_installed():
        install_VibronicToolkit()

    setup_info = dict(
        name='pibronic',
        version=version,
        python_requires='~={}.{}'.format(*REQUIRED_PYTHON),
        url='https://github.com/ngraymon/pibronic',
        author='Neil Raymond',
        author_email='neil.raymond@uwaterloo.ca',
        description='Quantum Mechanical Computational Package',
        long_description=read_md('README.md'),
        license='MIT',
        packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
        include_package_data=True,
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Chemistry',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        keywords='path_integral_monte_carlo quantum_mechanics chemistry vibronic',
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=[
            'numpy>=1.12',
            'scipy>=1.0.0',
            'matplotlib>=2.1.2',
            'parse==1.8.2',
        ],
        extras_require={
            'dev': ['check-manifest'],
            'test': ['coverage', 'pytest'],
            # electronic structure install
            'es': ['fortranformat==0.2.5',],
        },
    )

    setup(**setup_info)


if __name__ == '__main__':
    if julia_is_installed():
        install_VibronicToolkit()
    setup_package()
