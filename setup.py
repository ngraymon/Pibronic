"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# refer to https://docs.python.org/3/distutils/setupscript.html#distutils-installing-scripts

import subprocess
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


# verify that the right version of julia is installed
def julia_is_installed():
    p = subprocess.run(["julia", "--version"], stdout=subprocess.PIPE, universal_newlines=True)
    if p.stdout.startswith("julia version"):
        bits = p.stdout.split(sep=".")
        # if the version is 1.X.X or 0.[6-9].X
        if int(bits[0][-1]) >= 1 or int(bits[1]) >= 6:
            print("It appears julia 0.6.X or higher is installed")
            return True
        else:
            print("It appears julia is installed but I0.6.X or higher is installed")

    else:
        print("Julia is not installed and so VibronicToolkit cannot be installed",
              "some features such as: analytic/sos/trotter will not be available")
    return False


# verify that the right version of julia is installed
def install_VibronicToolkit():
    file_path = path.expanduser('~/.temp.jl')

    url = "https://github.com/ngraymon/VibronicToolkit.jl.git"
    package_name = "VibronicToolkit-Integrated"
    branch_name = "integrated"

    # should add a try catch so that julia doesn't error out if the package is already installed?
    # Pkg.installed("VibronicToolkit-Integrated")

    # This approach is functional but there is probably a better/cleaner way to do this
    contents = f'Pkg.clone("{url:s}", "{package_name:s}")\n'
    contents += f'Pkg.checkout("{package_name:s}", {branch_name:s})\n'

    with open(file_path, 'w') as file:
        file.write(contents)

    subprocess.run(['julia', file_path])
    subprocess.run(['rm', file_path])
    # we trust that the install was successful
    # although it would be nice to have a clean way to confirm the install was successful
    return


# use Pandoc magic to convert Markdown to RST for uploading to PyPi
def read_md(filename):
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, filename), encoding='utf-8') as f:
        try:
            from pypandoc import convert_text
            return convert_text(f.read(), 'rst', format="md")
        except ImportError:
            print("warning: pypandoc module not found, could not convert Markdown to RST")
            return f.read()


def setup_package():
    """ wapper for setup() so that we can check if julia is installed first """

    if julia_is_installed():
        install_VibronicToolkit()

    setup(
        name='pibronic',

        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version='0.2.0.dev1',

        description='Quantum Mechanical Computational Package',
        long_description=read_md('README.md'),
        # long_description=long_description,

        # The project's main homepage.
        url='https://github.com/ngraymon/pibronic',

        # Author details
        author='Neil Raymond',
        author_email='neil.raymond@uwaterloo.ca',

        # Choose your license
        license='MIT',

        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Chemistry',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            # 'Programming Language :: Python :: 3 :: Only',
            # 'Programming Language :: Python :: 3.3',
            # 'Programming Language :: Python :: 3.4',
            # 'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],

        # What does your project relate to?
        keywords='path_integral_monte_carlo quantum_mechanics chemistry vibronic',

        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=find_packages(exclude=['contrib', 'docs', 'tests', 'dev']),

        # Alternatively, if you want to distribute just a my_module.py, uncomment
        # this:
        #   py_modules=["my_module"],

        # the c extensions
        # Extension(
        #             'foo', ['foo.c'],
        #             include_dirs=['include'],
        #             library_dirs=['/usr/X11R6/lib'],
        #             libraries=['X11', 'Xt']
        #             )

        # installing scripts
        # scripts=[   'scripts/pbj',
        #             'scripts/pbj',
        #             ]

        # List run-time dependencies here.  These will be installed by pip when
        # your project is installed. For an analysis of "install_requires" vs pip's
        # requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=[
            'numpy>=1.12',
            'scipy>=1.0.0',
            'matplotlib>=2.1.2',
            'fortranformat==0.2.5',
            'parse==1.8.2',
            # '',
        ],

        python_requires='~=3.6',

        # List additional groups of dependencies here (e.g. development
        # dependencies). You can install these using the following syntax,
        # for example:
        # $ pip install -e .[dev,test]
        extras_require={
            'dev': ['check-manifest'],
            'test': ['coverage', 'pytest'],
        },

        # If there are data files included in your packages that need to be
        # installed, specify them here.  If using Python 2.6 or less, then these
        # have to be included in MANIFEST.in as well.
        # package_data={
        #     'sample': ['package_data.dat'],
        # },

        # Although 'package_data' is the preferred approach, in some case you may
        # need to place data files outside of your packages. See:
        # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
        # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
        # data_files=[('my_data', ['data/data_file'])],

        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        # entry_points={
        #     'console_scripts': [
        #         'sample=sample:main',
        #     ],
        # },
    )


if __name__ == '__main__':
    setup_package()
