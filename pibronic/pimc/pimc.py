# pimc.py

# system imports
import itertools as it
from functools import partial
# import cProfile
import json
# import time
# import sys
import os

# third party imports
import scipy.linalg
import numpy as np
from numpy import newaxis as NEW
from numpy import float64 as F64

# local imports
from ..log_conf import log
from .. import constants
from ..constants import hbar
from ..data import file_structure
from ..data import file_name  # do we need this?
# from ..data import vibronic_model_io as vIO
# from ..data.vibronic_model_keys import VibronicModelKeys as VMK
from ..vibronic import vIO, VMK
# from ..server import job_boss
from ..server.server import ServerExecutionParameters as SEP

"""
For future thought:
see - https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array/5837352#5837352
and - https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

Will have to think carefully about recording the random inputs that are sampled in a robust and cohesive manner
"""

__all__ = [
           "ModelClass",
           "ModelVibronic",
           "ModelVibronicPM",
           "ModelSampling",
           "BoxData",
           "BoxDataPM",
           "BoxResult",
           "BoxResultPM",
           "block_compute",
           "block_compute_pm",
           ]


float_tolerance = 1e-23

""" TODO - eventually this should be replaced so that each time a sample is drawn
a new seed is generated and stored in the results file indexed with the fraction it generated
"""
np.random.seed()  # random
# np.random.seed(232942)  # pick our seed


class TemperatureDependentClass:
    """store temperture dependent constants here"""
    def __init__(self, model, tau):
        # construct the coth and csch tensors
        omega = np.broadcast_to(model.omega, model.size['AN'])
        self.cothAN = np.tanh(hbar*tau*omega)**(-1.)
        self.cschAN = np.sinh(hbar*tau*omega)**(-1.)

        # construct different sized tensors for efficiency purposes
        self.cothANP = self.cothAN.copy().reshape(*self.cothAN.shape, 1)
        self.cschANP = self.cschAN.copy().reshape(*self.cschAN.shape, 1)

        self.cothBANP = self.cothAN.copy().reshape(1, *self.cothAN.shape, 1)
        self.cschBANP = self.cschAN.copy().reshape(1, *self.cschAN.shape, 1)

        # this is the constant prefactor that doesn't depend on sampled co-ordinates
        energy = np.diag(model.energy) if len(model.energy.shape) > 1 else model.energy
        tilde_energy = energy + model.delta_weight  # this is \tilde{E} from equation 34 on page 4
        prefactor = np.broadcast_to(tilde_energy, model.size['BPA'])
        self.omatrix_prefactor = np.exp(-tau * prefactor.copy())
        # this prefactor is F^P in equation 41 on page 5 where F is defined in equation 32
        prefactor = np.broadcast_to(self.cschAN, model.size['BPAN'])
        self.omatrix_prefactor *= np.prod(prefactor.copy(), axis=3)**0.5

        # note that there is no sqrt(1/2*pi) because it factors out of the numerator and denominator

        # cache for the Omatrix on each block loop
        self.omatrix = np.zeros(model.size['BPAA'])
        # cache for the Omatrix scaling factor on each block loop
        self.omatrix_scaling = np.empty(model.size['BP'])
        return


class ModelClass:
    """information describing a quantum mechanical system"""
    states = 0
    modes = 0
    omega = None
    energy = None
    linear = None
    quadratic = None
    cubic = None
    quartic = None

    def __init__(self, states=1, modes=1):
        """x"""
        self.states = states
        self.modes = modes
        self.state_range = range(states)
        self.mode_range = range(modes)
        return

    @classmethod
    def from_json_file(cls, path):
        """constructor wrapper"""
        with open(path, mode='r', encoding='UTF8') as target_file:
            file_data = target_file.read()
        return cls.load_model(cls, file_data)

    def load_model(self, path):
        """x"""
        # I think this fails if the list elements are multidimensional numpy arrays
        # carefully check this
        # lazy way of assuming that if one array is empty this is the first time
        if type(None) in map(type, [self.omega, self.energy, self.linear, self.quadratic]):
            kwargs = vIO.load_model_from_JSON(path)
            self.energy = kwargs[VMK.E]
            self.omega = kwargs[VMK.w]
            self.linear = kwargs[VMK.G1]
            A = self.states
            N = self.modes
            shape = vIO.model_shape_dict(A, N)
            self.linear = kwargs[VMK.G1] if VMK.G1 in kwargs else np.zeros(shape[VMK.G1], dtype=F64)
            self.quadratic = kwargs[VMK.G2] if VMK.G2 in kwargs else np.zeros(shape[VMK.G2], dtype=F64)
        else:
            kwargs = {}
            kwargs[VMK.N] = self.modes
            kwargs[VMK.A] = self.states
            kwargs[VMK.E] = self.energy
            kwargs[VMK.w] = self.omega
            kwargs[VMK.G1] = self.linear
            kwargs[VMK.G2] = self.quadratic
            vIO.load_model_from_JSON(path, kwargs)
        # should we update the states and modes after loading the model?
        self.modes = kwargs[VMK.N]
        self.states = kwargs[VMK.A]
        self.energy = kwargs[VMK.E]
        self.omega = kwargs[VMK.w]
        self.linear = kwargs[VMK.G1]
        self.quadratic = kwargs[VMK.G2]
        return


class ModelVibronic(ModelClass):
    """stores information about the system of interest"""
    def __init__(self, data):
        super().__init__(data.states, data.modes)
        self.size = data.size
        self.beta = data.beta
        self.tau = data.tau

        # model paramters
        self.omega = np.zeros(self.size['N'], dtype=F64)
        self.energy = np.zeros(self.size['AA'], dtype=F64)
        self.linear = np.zeros(self.size['NAA'], dtype=F64)
        self.quadratic = np.zeros(self.size['NNAA'], dtype=F64)

        # sampling parameters
        self.delta_weight = np.zeros(self.size['A'], dtype=F64)
        # self.state_weight = np.zeros(self.size['A'], dtype=F64)
        self.state_shift = np.zeros(self.size['AN'], dtype=F64)
        return

    def compute_linear_displacement(self, data):
        """compute the energy shift equivalent to a linear displacement"""
        for a in range(data.states):
            # UNDO - we would have to untransform the prefactor here
            self.delta_weight[a] = -0.5 * (self.linear[:, a, a]**2. / self.omega).sum(axis=0)
        return

    def compute_weight_for_each_state(self, modified_energy):
        """these are the weights for the oscillators associated with each state"""
        assert False, "you shall not pass"  # haha it doesn't do anything anymore
        self.state_weight = -self.beta * (modified_energy + self.delta_weight)
        self.state_weight = np.exp(self.state_weight)
        self.state_weight /= 2. * np.prod(np.sinh((self.beta * self.omega) / 2.))
        # normalize the weights
        self.state_weight /= self.state_weight.sum()
        return

    def optimize_energy(self):
        """ shift the energy to reduce the order of the raw partition function value """

        # -- CURRENTLY DOES NOTHING -- #
        # So the problem here is that if you scale the energy you have to affect
        # the output because the initial values are just different
        # self.energyShift = np.nanmin(np.diagonal(self.energy)
        #                                    + self.delta_weight)
        # only shift the diagonal of the energy
        # for a in range(self.states):
        #     self.energy[a,a] -= self.energyShift

        return

    def initialize_TDP_object(self):
        """creates the TemperatureDependentClass object"""
        self.const = TemperatureDependentClass(self, self.tau)
        return

    def finish_folding_in_terms(self, data):
        """ set the terms we 'folded in' to zero """
        for a in range(data.states):
            # we 'fold' the linear term into the harmonic oscillator
            self.state_shift[a, :] = -self.linear[:, a, a] / self.omega
            # then we remove it from the calculation
            self.linear[:, a, a] = 0.0

            # zero energy
            self.energy[a, a] = 0.0
        return

    def precompute(self, data):
        """precompute some constants"""

        self.compute_linear_displacement(data)

        # duplicate the diagonal and then zero it for later use in the M matrix
        energyDiag = np.diag(self.energy).copy()

        self.optimize_energy()

        # self.compute_weight_for_each_state(energyDiag)

        self.initialize_TDP_object()

        self.finish_folding_in_terms(data)
        return


class ModelVibronicPM(ModelVibronic):
    """plus minus version of ModelVibronic"""
    delta_beta = 0.0
    beta_plus = 0.0
    beta_minus = 0.0
    tau_plus = 0.0
    tau_minus = 0.0

    def __init__(self, data):
        super().__init__(data)
        self.delta_beta = data.delta_beta
        return

    def initialize_TDP_object(self):
        """creates the TemperatureDependentClass object"""

        super().initialize_TDP_object()

        self.const_plus = TemperatureDependentClass(self, self.tau_plus)
        self.const_minus = TemperatureDependentClass(self, self.tau_minus)
        return

    # precompute some constants
    def precompute(self, data):
        # store extra constants
        self.beta_plus = self.beta + self.delta_beta
        self.beta_minus = self.beta - self.delta_beta
        self.tau_plus = self.beta_plus / data.beads
        self.tau_minus = self.beta_minus / data.beads

        super().precompute(data)
        return


class ModelSampling(ModelClass):
    """stores information about the system of interest"""
    def __init__(self, data):
        # copy the sizes of the parameters
        self.param_dict = data.param_dict.copy()
        self.size_list = data.size_list.copy()
        self.tau = data.tau
        self.beta = data.beta
        return

    def load_model(self, filePath):
        newStates, sameModes = vIO.extract_dimensions_of_sampling_model(path=filePath)

        # replace the vibronic models state size with rho's state size
        self.param_dict['A'] = self.states = newStates
        self.modes = sameModes

        # construct 'size' tuples
        self.size = {}
        for key in self.size_list:
            self.size[key] = tuple([self.param_dict[letter] for letter in key])

        # model paramters
        self.omega = np.zeros(self.size['N'], dtype=F64)
        self.energy = np.zeros(self.size['A'], dtype=F64)
        self.linear = np.zeros(self.size['NA'], dtype=F64)
        # in this case we might need to rethink the load_sample_from_JSON - here we provide quadratic when they are zero
        self.quadratic = np.zeros(self.size['NNA'], dtype=F64)

        # sampling parameters
        self.delta_weight = np.zeros(self.size['A'], dtype=F64)
        self.state_weight = np.zeros(self.size['A'], dtype=F64)
        self.state_shift = np.zeros(self.size['AN'], dtype=F64)
        self.cc_samples = np.zeros(self.size['BNP'], dtype=F64)

        # this brings up the good point that we might want to load a JSON file without providing the number of modes and surfaces
        kwargs = {}
        kwargs[VMK.N] = self.modes
        kwargs[VMK.A] = self.states
        kwargs[VMK.E] = self.energy
        kwargs[VMK.w] = self.omega
        kwargs[VMK.G1] = self.linear
        # print(kwargs[VMK.G1], "\n", self.linear)
        kwargs[VMK.G2] = self.quadratic
        vIO.load_sample_from_JSON(filePath, kwargs)
        self.modes = kwargs[VMK.N]
        self.states = kwargs[VMK.A]
        self.energy = kwargs[VMK.E]
        self.omega = kwargs[VMK.w]
        self.linear = kwargs[VMK.G1]
        # print(kwargs[VMK.G1], "\n", self.linear)
        self.quadratic = kwargs[VMK.G2]
        return

    def draw_sample(self, sample_view):
        """Generates collective co-ordinates and stores them in self.cc_samples with dimensions BNP"""
        # collective co-ordinate samples
        self.cc_samples = np.random.normal(
                                    loc=self.sample_means,
                                    scale=self.standard_deviation[sample_view],
                                    size=self.size['BNP'],
                                    )
        return

    def compute_linear_displacement(self):
        """compute the energy shift equivalent to a linear displacement"""
        self.delta_weight = -0.5 * (self.linear**2. / self.omega[:, NEW]).sum(axis=0)
        return

    def compute_weight_for_each_state(self):
        print(self.beta, self.energy, self.omega, sep='\n')
        """these are the weights for the oscillators associated with each state"""
        self.state_weight = np.exp(-self.beta * (self.energy + self.delta_weight))
        # self.state_weight /= 2. * np.prod(np.sinh((self.beta * self.omega) / 2.))

        # we believe the factor of 2 on the outside cancels out in equation 49.
        self.state_weight /= np.prod(np.sinh((self.beta * self.omega) / 2.))

        # normalize the weights
        self.state_weight /= self.state_weight.sum()
        print("weights", self.state_weight)
        return

    def optimize_energy(self):
        """ shift the energy to reduce the order of the raw partition function value """

        # -- CURRENTLY DOES NOTHING -- #
        # So the problem here is that if you scale the energy you have to affect
        # the output because the initial values are just different
        # self.energyShift = np.nanmin(np.diagonal(self.energy)
        #                                    + self.delta_weight)
        # only shift the diagonal of the energy
        # for a in range(self.states):
        #     self.energy[a,a] -= self.energyShift

        return

    def initialize_TDP_object(self):
        """creates the TemperatureDependentClass object"""
        self.const = TemperatureDependentClass(self, self.tau)
        return

    def finish_folding_in_terms(self):
        """ set the terms we 'folded in' to zero """
        for a in range(self.states):
            # we 'fold' the linear term into the harmonic oscillator
            self.state_shift[a, :] = -self.linear[:, a] / self.omega
            # then we remove it from the calculation
            self.linear[:, a] = 0.0
        return

    def compute_sampling_constants(self, data):

        # generate random surfaces to draw samples from
        self.sample_sources = np.random.choice(range(self.states),
                                               size=self.size['X'],
                                               p=self.state_weight
                                               )

        # the ordered offsets for multiple surfaces
        self.sample_shift = self.state_shift[self.sample_sources, :]
        # calculate the means of each gaussian

        # inverse_covariance_matrix = 2. * coth_tensor[:, :, NEW] - sch_tensor[:, :, NEW] * O_eigvals[NEW, NEW, :]
        # inverse_covariance_matrix = 2. * self.const.coth[..., NEW] - self.const.csch[..., NEW] * data.circulant_eigvals[NEW, NEW, ...]

        self.inverse_covariance = (2. * self.const.cothANP
                                   - self.const.cschANP * data.circulant_eigvals)
        # print(data.circulant_eigvects.T)
        # print(  "Mode 1 ",
        #         self.inverse_covariance[0,0,-1],
        #         np.sqrt(self.inverse_covariance[0,0,-1]),
        #         np.divide(1., np.sqrt(self.inverse_covariance[0,0,-1])),
        #         np.divide(1., np.sqrt(self.inverse_covariance[0,0,-1])) / np.sqrt(self.size['P'][0]),
        #         6*(np.divide(1., np.sqrt(self.inverse_covariance[0,0,-1])) / np.sqrt(self.size['P'][0])),
        #         np.divide(1., np.sqrt(self.inverse_covariance[0,0,-1])) / np.sqrt(self.size['P'][0]/ 20),
        #         sep="\n",
        #     )
        # print(  "Mode 2 ",
        #         self.inverse_covariance[0,1,-1],
        #         np.sqrt(self.inverse_covariance[0,1,-1]),
        #         np.divide(1., np.sqrt(self.inverse_covariance[0,1,-1])),
        #         np.divide(1., np.sqrt(self.inverse_covariance[0,1,-1])) / np.sqrt(self.size['P'][0]),
        #         6*(np.divide(1., np.sqrt(self.inverse_covariance[0,1,-1])) / np.sqrt(self.size['P'][0])),
        #         np.divide(1., np.sqrt(self.inverse_covariance[0,1,-1])) / np.sqrt(self.size['P'][0] / 20),
        #         sep="\n",
        #     )
        # self.self.inverse_covariance -= self.const.cschANP * data.circulant_eigvals
        self.standard_deviation = np.sqrt(1. / self.inverse_covariance[self.sample_sources, ...])
        self.sample_means = np.zeros(self.size['BNP'], dtype=F64)
        return

    # precompute some constants
    def precompute(self, data):
        """precompute some constants"""

        # HACKS - TEMPORARY CHANGE
        self.compute_linear_displacement()

        self.optimize_energy()

        self.compute_weight_for_each_state()

        self.initialize_TDP_object()

        self.finish_folding_in_terms()

        self.compute_sampling_constants(data)
        return


class BoxData:
    """use this to pass execution paramters back and forth between methods"""
    block_size = 0
    samples = 0
    blocks = 0

    # rho_states = 0
    states = 0
    modes = 0

    temperature = 0.0
    beads = 0

    beta = 0.0
    tau = 0.0
    delta_beta = 0.0

    id_data = 0
    id_rho = 0

    # instance of the ModelVibronic class
    # holds all parameters associated with the model
    vib = None
    path_vib_model = ""

    # instance of the ModelSampling class
    # holds all parameters associated with the model
    rho = None
    path_rho_model = ""

    # it would probably be better to shuffle this json passing black magic into job boss
    # perhaps job boss modifies the load_json function before calling it?
    # or maybe one of the optional paramters to load_json is the replacement symbol which by default is a colon?

    # we need to use SOME symbol(semicolon for example)
    # to allow the string to be treated as a SINGLE environment variable
    # slurm uses commas as the delimiters for its environment variable list
    _COMMA_REPLACEMENT = ";"
    _SEPARATORS = (_COMMA_REPLACEMENT, ':')

    # the hash values for the vibronic model and the gaussian mixture distribution
    hash_vib = None
    hash_rho = None

    @classmethod
    def from_FileStructure(cls, FS):
        """constructor wrapper"""
        data = cls()
        data.id_data = FS.id_data
        data.id_rho = FS.id_rho

        data.path_vib_model = FS.path_vib_model
        data.path_rho_model = FS.path_rho_model

        A, N = vIO.extract_dimensions_of_coupled_model(FS)
        data.states = A
        data.modes = N

        return data

    @classmethod  # this feels unnecessary
    def build(cls, id_data, id_rho):
        """constructor wrapper"""
        data = cls()
        data.id_data = id_data
        data.id_rho = id_rho
        return data

    @classmethod
    def from_json_file(cls, path):
        """constructor wrapper"""
        with open(path, mode='r', encoding='UTF8') as target_file:
            file_data = target_file.read()
        return cls.from_json_string(cls, file_data)

    @classmethod
    def from_json_string(cls, json_str):
        """constructor wrapper"""
        data = cls()
        data.load_json_string(json_str)
        return data

    def __init__(self):
        return

    @classmethod
    def json_serialize(cls, params):
        json_str = json.dumps(params, separators=cls._SEPARATORS)
        return json_str

    def encode_self(self, params=None):
        """encodes json str with member values or given params"""
        log.debug("Encoding data to JSON")

        if params is None:
            log.flow("params was none")
            params = {
                SEP.X: self.samples,
                SEP.nBlk: self.blocks,
                SEP.A: self.states,
                SEP.P: self.beads,
                SEP.N: self.modes,
                SEP.T: self.temperature,
                SEP.BlkS: self.block_size,
                SEP.dB: self.delta_beta,
                SEP.D: self.id_data,
                SEP.R: self.id_rho,
                SEP.beta: self.beta,
                SEP.tau: self.tau,
                # "s": self.hash_vib,
                # "x": self.hash_rho,
            }
        log.flow(params)
        return self.json_serialize(params)

    # is this still used?
    def load_json_string(self, json_str):
        """decodes the json_str and sets member parameters"""
        log.debug("Decoding JSON obj")
        json_str = json_str.replace(self._COMMA_REPLACEMENT, ",")
        params = json.loads(json_str)
        # We should only set the value if they are not none?

        # replace with dictionary update?
        # def setVar(self, var):
        # for key, value in var.items():
        # setattr(self, key, value)
        self.samples = params[SEP.X.value]
        self.blocks = params[SEP.nBlk.value]
        self.states = params[SEP.A.value]
        self.beads = params[SEP.P.value]
        self.modes = params[SEP.N.value]
        self.temperature = params[SEP.T.value]
        self.block_size = params[SEP.BlkS.value]
        self.id_data = params[SEP.D.value]
        self.id_rho = params[SEP.R.value]
        # self.hash_vib = params[]
        # self.hash_rho = params[]

        # dumb hacks
        if SEP.beta.value in params.keys():
            self.beta = params[SEP.beta.value]
        else:
            self.beta = 1.0 / (constants.boltzman * self.temperature)
        # dumb hacks
        if SEP.tau.value in params.keys():
            self.tau = params[SEP.tau.value]
        else:
            self.tau = self.beta / self.beads
        # dumb hacks
        if SEP.dB.value in params.keys():
            if type(params[SEP.dB.value]) is float:
                self.delta_beta = params[SEP.dB.value]

        # fill the paths
        FS = file_structure.FileStructure(params["path_root"], self.id_data, self.id_rho)
        self.path_vib_model = FS.path_vib_model
        self.path_rho_model = FS.path_rho_model

        FS.generate_model_hashes()
        self.hash_vib = FS.hash_vib
        self.hash_rho = FS.hash_rho

        for k, v in params.items():
            log.debug(type(v), k, v)

        return

    def draw_sample(self, sample_view):
        """Draws samples from the distribution rho -
        the rho object fills its cc_samples parameter with collective co-ordinates
        which will be transformed to bead dependent co-ordinates by self.transform_sampled_coordinates()"""
        self.rho.draw_sample(sample_view)
        return

    def generate_random_R_values(self, result, storage_array, sample_view):
        """Randomly generates R values that have no relation to the distribution rho or g
        they only have the correct dimensions BANP and are shifted appropriately """
        # generate sample points in dimensionless co-ordinates R
        storage_array[sample_view, :, :] = np.random.random(size=self.rho.size['BNP'])
        np.copyto(self.qTensor, storage_array[sample_view, NEW, ...])

        # we transform the dimensionless co-ordinates to surface dependent co-ordinates q = R - d
        self.qTensor -= self.vib.state_shift[NEW, :, :, NEW]
        return

    def transform_sampled_coordinates(self, sample_view):
        """transform from collective co-ordinates to bead dependent co-ordinates"""
        # self.qTensor = np.broadcast_to(np.einsum('ab,ijb->ija', self.circulant_eigvects, self.cc_samples), self.size['XANP'])
        # self.qTensor = np.einsum('ab,ijb->ija', self.circulant_eigvects, self.cc_samples)[:, NEW, :, :]
        # self.qTensor = np.einsum('ab,ijb->ija', self.circulant_eigvects, self.cc_samples)[:, NEW, ...]
        self.qTensor[:] = np.einsum('ab,ijb->ija',
                                    self.circulant_eigvects,
                                    self.rho.cc_samples,
                                    )[:, NEW, ...]

        # remove sample dependent normal mode displacement (from sampling model)
        self.qTensor += self.rho.sample_shift[sample_view, NEW, :, NEW]
        # add surface dependent normal mode displacement (from vibronic model)
        self.qTensor -= self.vib.state_shift[NEW, :, :, NEW]
        return

    def initialize_models(self):
        """x"""
        self.vib = ModelVibronic(self)
        self.vib.load_model(self.path_vib_model)
        self.vib.precompute(self)

        self.rho = ModelSampling(self)
        self.rho.load_model(self.path_rho_model)
        self.rho.precompute(self)
        return

    def preprocess(self):
        """x"""
        # for readability and clarity we use these letters
        self.param_dict = {'X': self.samples,
                           'A': self.states,
                           'N': self.modes,
                           'P': self.beads,
                           'B': self.block_size, }

        self.size_list = ['X', 'P', 'N', 'A', 'B', 'XP',
                          'BP', 'AN', 'NA', 'AA',
                          'BNP', 'BPA', 'BAA', 'XNP',
                          'NAA', 'NNA', 'ANP',
                          'BANP', 'BPAA', 'NNAA', 'BPAN', ]

        # construct 'size' tuples
        self.size = {}
        for key in self.size_list:
            self.size[key] = tuple([self.param_dict[letter] for letter in key])

        # compute constants
        self.beta = 1.0 / (constants.boltzman * self.temperature)
        self.tau = self.beta / self.beads

        # where we store the transformed samples
        self.qTensor = np.zeros(self.size['BANP'], dtype=F64)

        # storage for the numerator calculation
        self.coupling_matrix = np.zeros(self.size['BPAA'])
        self.coupling_eigvals = np.empty(self.size['BPA'])
        self.coupling_eigvects = np.empty(self.size['BPAA'])
        self.M_matrix = np.empty(self.size['BPAA'])
        self.numerator = np.zeros(self.size['BAA'])

        # construct the circulant matrix
        assert self.beads >= 3, "circulant matrix requires 3 or more beads"  # hard check
        defining_vector = [0, 1] + [0]*(self.beads-3) + [1]
        self.circulant_matrix = scipy.linalg.circulant(defining_vector)

        (self.circulant_eigvals,
         self.circulant_eigvects
         ) = np.linalg.eigh(self.circulant_matrix, UPLO='L')

        self.initialize_models()
        return


class BoxDataPM(BoxData):
    """plus minus version of BoxData"""
    delta_beta = 0.0
    beta_plus = 0.0
    beta_minus = 0.0
    tau_plus = 0.0
    tau_minus = 0.0

    def __init__(self, delta_beta=None):
        # -TODO -
        # consider removing the requirement of providing a delta_beta
        # could use a default value from constant module
        # with the ability to optionally override
        if delta_beta is None:
            delta_beta = constants.delta_beta
        self.delta_beta = delta_beta
        super().__init__()
        return

    def initialize_models(self):
        """"""
        self.vib = ModelVibronicPM(self)
        self.vib.load_model(self.path_vib_model)
        self.vib.precompute(self)

        self.rho = ModelSampling(self)
        self.rho.load_model(self.path_rho_model)
        self.rho.precompute(self)
        return

    def preprocess(self):
        """"""
        # compute extra constants
        self.beta_plus = self.beta + self.delta_beta
        self.beta_minus = self.beta - self.delta_beta
        self.tau_plus = self.beta_plus / self.beads
        self.tau_minus = self.beta_minus / self.beads

        # do the usual work
        super().preprocess()
        return


class BoxResult:
    """use this to pass results back and forth between methods"""

    id_job = None
    # TODO - temporary solution, the preference is to be able to loosely create the object and provide the necessary information later
    hash_vib = None
    hash_rho = None

    key_list = ["number_of_samples", "s_rho", "s_g"]

    @classmethod
    def read_number_of_samples(cls, path_full):
        """x"""
        with np.load(path_full, mmap_mode='r') as data:
            return data["number_of_samples"]

    @classmethod
    def verify_result_keys_are_present(cls, path, fileObj):
        """x"""
        for k in cls.key_list:
            if k not in fileObj.keys():
                s = "Expected key ({:s}) not present in result file\n{:s}\n"
                raise AssertionError(s.format(k, path))

    @classmethod
    def result_keys_are_present_in(cls, iterable):
        """ x """
        for k in cls.key_list:
            if k not in iterable:
                return False
        return True

    def initialize_arrays(self):
        self.scaled_g = np.full(self.samples, np.nan, dtype=F64)
        self.scaled_rho = np.full(self.samples, np.nan, dtype=F64)
        return

    def __init__(self, data=None, X=None):
        """x"""
        if data is not None:
            self.partial_name = partial(file_name.pimc().format, P=data.beads, T=data.temperature)
            self.samples = data.samples
            self.hash_vib = data.hash_vib
            self.hash_rho = data.hash_rho
        elif X is not None:
            self.samples = X
        else:
            self.samples = 0
            log.debug("The BoxResult object has been initialized with 0 samples this could be an issue?")
            # raise AssertionError("data or X must be provided to BoxResult __init__")

        # assert data is not None, "we can't take data's hash values if they don't exist!"
        # FS.generate_model_hashes()  # TODO - possibly remove this in the future?
        # TODO - better way of passing the hashes around
        # self.hash_vib = data.hash_vib
        # self.hash_rho = data.hash_rho
        self.initialize_arrays()
        return

    def compute_path_to_file(self):
        """ used by BoxResultPM as well """
        if self.samples is 0:
            raise AssertionError("{:s} still has 0 samples - this should not happen".format(self.__class__.__name__))
        elif self.id_job is not None:
            self.name = self.partial_name(J=int(self.id_job))  # why was this a str?
        else:
            """ TODO - this could be dangerous on the server if a BoxResult object is created
            but not assigned a job id - need to create a test to prevent this from happening
            """
            # should be 0 or the last number + 1
            self.name = self.partial_name(J=0)
            if False:  # search for other job id's
                old_j = 90  # placeholder
                self.name = self.partial_name(J=old_j+1)

        # result_view = slice(0, number_of_samples)
        return os.path.join(self.path_root, self.name)

    def save_results(self, number_of_samples):
        """x"""
        path = self.compute_path_to_file()

        assert self.hash_vib is not None and self.hash_rho is not None, "we save hash values if they don't exist!"
        # save raw data points
        np.savez(path,
                 # TODO - fix the hash situation - this is hacky
                 hash_vib=self.hash_vib,
                 hash_rho=self.hash_rho,
                 number_of_samples=self.samples,
                 s_rho=self.scaled_rho,
                 s_g=self.scaled_g,
                 # s_g=self.scaled_g[result_view],
                 # s_rho=self.scaled_rho[result_view],
                 )
        return

    def load_results(self, path):
        """ load results from one file"""
        with np.load(path, mmap_mode="r") as data:
            self.__class__.verify_result_keys_are_present(path, data)
            # TODO - check hashes here? - or do we assume they've already been checked?
            # BoxResult.verify_hashes_are_valid()
            if self.samples is 0:
                self.samples = data["number_of_samples"]
            elif data["number_of_samples"] is not self.samples:
                raise AssertionError("BoxResult has a different number of samples that the input file - this should not happen")

            self.initialize_arrays()
            self.scaled_g = data["s_g"]
            self.scaled_rho = data["s_rho"]
        return

    def load_multiple_results(self, list_of_paths, desired_number_of_samples=None):
        """ load results from more than one file
        the optional argument desired_number_of_samples can be provided
        the method will then try to load as many samples UPTO the desired_number_of_samples and no more"""
        number_of_samples = 0
        assert not len(list_of_paths) == 0, "list_of_paths cannot be empty"

        list_of_bad_paths = []

        for path in list_of_paths:
            # should verify path is correct?
            with np.load(path, mmap_mode="r") as data:
                """ TODO - design decision choice here
                This implementation currently prints out a debug message and continues execution assuming it's job is to scan through and load as many files as possible.
                For each path that doesn't have the data we expect it to have we will remove it from the list.
                Another alternative would be to raise an AssertionError if the file it is trying to load does not have the appropriate keys, which is a good approach.
                """
                # self.__class__.verify_result_keys_are_present(path, data)

                # TODO - check hashes here? - or do we assume they've already been checked?
                # BoxResult.verify_hashes_are_valid()
                # should verify file is not empty?
                if self.__class__.result_keys_are_present_in(data.keys()):
                    number_of_samples += data["number_of_samples"]
                else:
                    list_of_bad_paths.append(path)

        # first lets make sure that we had 1 or more good paths
        assert not set(list_of_paths) == set()

        # okay now remove all the bad paths!
        list_of_paths = list(set(list_of_paths) - set(list_of_bad_paths))

        assert len(list_of_paths) > 0, "none of the provided paths were good, i.e. none of them had all the required keys {}".format(self.key_list)

        assert number_of_samples is not 0, "number of samples should have changed"

        if desired_number_of_samples is None:
            desired_number_of_samples = number_of_samples
        else:
            if number_of_samples >= desired_number_of_samples:
                number_of_samples = desired_number_of_samples
            elif number_of_samples < desired_number_of_samples:
                print("We found less samples than were requested!!")
                # TODO - choose what the correct procedure in this case is, for now we shall not raise an error

        self.samples = number_of_samples
        self.initialize_arrays()

        start = 0
        for path in list_of_paths:
            if start >= desired_number_of_samples:
                break
            with np.load(path) as data:
                length = min(data["number_of_samples"], desired_number_of_samples)
                assert not np.any(data["s_rho"][0:length] == 0.0), "Zeros in the denominator"
                self.scaled_g[start:start+length] = data["s_g"][0:length]
                self.scaled_rho[start:start+length] = data["s_rho"][0:length]
                start += length
        return


class BoxResultPM(BoxResult):
    """plus minus version of BoxResult"""

    key_list = ["s_gP", "s_gM"] + BoxResult.key_list

    @classmethod
    def verify_result_keys_are_present(cls, path, fileObj):
        """ x """
        super().verify_result_keys_are_present(path, fileObj)
        for k in cls.key_list:
            if k not in fileObj.keys():
                s = "Expected key ({:s}) not present in result file\n{:s}\n"
                raise AssertionError(s.format(k, path))

    @classmethod
    def result_keys_are_present_in(cls, iterable):
        """ x """
        for k in cls.key_list:
            if k not in iterable:
                return False
        return True and super().result_keys_are_present_in(iterable)

    def initialize_arrays(self):
        super().initialize_arrays()
        self.scaled_gofr_plus = np.full(self.samples, np.nan, dtype=F64)
        self.scaled_gofr_minus = np.full(self.samples, np.nan, dtype=F64)
        return

    def __init__(self, data=None, X=None):
        """x"""
        super().__init__(data, X)
        return

    def save_results(self, number_of_samples):
        """x"""
        path = self.compute_path_to_file()

        assert self.hash_vib is not None and self.hash_rho is not None, "we save hash values if they don't exist!"
        # save raw data points
        np.savez(path,
                 # TODO - fix the hash situation - this is hacky
                 hash_vib=self.hash_vib,
                 hash_rho=self.hash_rho,
                 number_of_samples=self.samples,
                 s_rho=self.scaled_rho,
                 s_g=self.scaled_g,
                 s_gP=self.scaled_gofr_plus,
                 s_gM=self.scaled_gofr_minus,
                 )
        return

    def load_results(self, path):
        """x"""
        super().load_results(path)
        # this could be more efficient, since we open the file twice
        with np.load(path) as data:
            self.scaled_gofr_plus = data["s_gP"]
            self.scaled_gofr_minus = data["s_gM"]
        return

    def load_multiple_results(self, list_of_paths, desired_number_of_samples=None):
        """ load results from more than one file
        the optional argument desired_number_of_samples can be provided
        the method will then try to load as many samples UPTO the desired_number_of_samples and no more"""
        number_of_samples = 0
        assert not len(list_of_paths) == 0, "list_of_paths cannot be empty"

        list_of_bad_paths = []

        try:
            for path in list_of_paths:
                # should verify path is correct?
                with np.load(path, mmap_mode="r") as data:
                    """ TODO - design decision choice here
                    This implementation currently prints out a debug message and continues execution assuming it's job is to scan through and load as many files as possible.
                    For each path that doesn't have the data we expect it to have we will remove it from the list.
                    Another alternative would be to raise an AssertionError if the file it is trying to load does not have the appropriate keys, which is a good approach.
                    """
                    # self.__class__.verify_result_keys_are_present(path, data)

                    # TODO - check hashes here? - or do we assume they've already been checked?
                    # BoxResult.verify_hashes_are_valid()
                    # should verify file is not empty?
                    if self.__class__.result_keys_are_present_in(data.keys()):
                        number_of_samples += data["number_of_samples"]
                    else:
                        list_of_bad_paths.append(path)
        except Exception as err:
            print("Did we get another mangled .npz file?\nIf numpy can't load the file because it is not a zip file then just delete the offending file and rerun the script")
            raise(err)

        # first lets make sure that we had 1 or more good paths
        assert not set(list_of_paths) == set()

        # okay now remove all the bad paths!
        list_of_paths = list(set(list_of_paths) - set(list_of_bad_paths))

        assert len(list_of_paths) > 0, "none of the provided paths were good, i.e. none of them had all the required keys {}".format(self.key_list)

        assert number_of_samples is not 0, "number of samples should have changed"

        if desired_number_of_samples is None:
            desired_number_of_samples = number_of_samples
        else:
            if number_of_samples >= desired_number_of_samples:
                number_of_samples = desired_number_of_samples
            elif number_of_samples < desired_number_of_samples:
                print("We found less samples than were requested!!")
                # TODO - choose what the correct procedure in this case is, for now we shall not raise an error

        self.samples = number_of_samples
        self.initialize_arrays()

        start = 0
        for path in list_of_paths:
            if start >= desired_number_of_samples:
                break
            with np.load(path) as data:
                length = min(data["number_of_samples"], desired_number_of_samples)
                assert not np.any(data["s_rho"][0:length] == 0.0), "Zeros in the denominator"
                self.scaled_g[start:start+length] = data["s_g"][0:length]
                self.scaled_rho[start:start+length] = data["s_rho"][0:length]
                self.scaled_gofr_plus[start:start+length] = data["s_gP"][0:length]
                self.scaled_gofr_minus[start:start+length] = data["s_gM"][0:length]
                start += length
        return


def pos_sym_assert(tensor):
    """raises error if provided tensor is not positive semi-definite"""

    # One method is to check if the matrix is positive semi definite within some tolerance
    isPositiveSemiDefinite = np.all(tensor > -float_tolerance)
    if not isPositiveSemiDefinite:
        s = "The Covariance matrix is not symmetric positive-semidefinite"
        raise AssertionError(s)

    # alternatively we can try to compute choleskys decomposition
    try:
        np.linalg.choleskys(tensor)
    except np.linalg.LinAlgError as e:
        s = "The Covariance matrix is not symmetric positive-semidefinite"
        raise AssertionError(s)

    return


def scale_o_matricies(scalingFactor, model_one, model_two):
    """divides the O matricies of the models by the scalingFactor"""
    model_one.omatrix /= scalingFactor[..., NEW, NEW]
    model_two.omatrix /= scalingFactor[..., NEW, NEW]
    return


def un_scale_o_matricies(scalingFactor, model_one, model_two):
    """multiples the O matricies of the models by the scalingFactor"""
    model_one.omatrix *= scalingFactor[..., NEW, NEW]
    model_two.omatrix *= scalingFactor[..., NEW, NEW]
    return


def build_scaling_factors(S12, model_one, model_two):
    """Calculates the individual and combined scaling factors for both provided models"""
    # compute the individual scaling factors
    model_one.omatrix_scaling[:] = np.amax(model_one.omatrix, axis=(2, 3))
    model_two.omatrix_scaling[:] = np.amax(model_two.omatrix, axis=(2, 3))

    # compute the combined scaling factor
    S12[:] = np.maximum(model_one.omatrix_scaling, model_two.omatrix_scaling)
    return


def build_o_matrix(data, model):
    """Calculates the O matrix of a model, storing the result inside the model object"""

    # name and select the views
    q1 = data.qTensor.view()
    q2 = np.roll(q1.view(), shift=-1, axis=3)

    coth = model.cothBANP.view()
    csch = model.cschBANP.view()

    # compute the omatrix
    o_matrix = -0.5 * np.sum(coth * (q1**2. + q2**2.) - 2.*csch*q1*q2, axis=2).swapaxes(1, 2)

    np.exp(o_matrix, out=o_matrix)

    for a in range(data.states):
        model.omatrix[:, :, a, a] = o_matrix[:, :, a]

    temp_states = model.omatrix.shape[2]
    # asserts that the tensor is diagonal along axis (2,3)
    for a, b in it.product(range(temp_states), range(temp_states)):
        if a == b:
            continue
        assert(np.all(model.omatrix[:, :, a, b] == 0.0))

    # this only works because omatrix is filled with zeros already
    model.omatrix *= model.omatrix_prefactor[..., NEW]
    return


def build_denominator(rho_model, outputArray, idx):
    """Calculates the state trace over the bead product of the o matricies of the rho model"""
    # outputArray[idx] = rho_model.omatrix.prod(axis=1).sum(axis=1)
    outputArray[idx] = rho_model.omatrix.prod(axis=1).trace(axis1=1, axis2=2)
    return


def diagonalize_coupling_matrix(data):

    # ------------------------------------------------------------------------
    # shift to surface independent co-ordinates
    data.qTensor += data.vib.state_shift[NEW, :, :, NEW]

    # build the coupling matrix
    # quadratic terms
    data.coupling_matrix[:] = np.einsum('acef, debc, abdf->afbc',
                                        data.qTensor,
                                        0.5*data.vib.quadratic,
                                        data.qTensor,
                                        # optimize='optimal',  # not clear if this is faster
                                        )
    # linear terms
    data.coupling_matrix += np.einsum('dbc, abdf->afbc',
                                      data.vib.linear,
                                      data.qTensor,
                                      # optimize='optimal',  # not clear if this is faster
                                      )
    # reference hamiltonian (energy shifts)
    data.coupling_matrix += data.vib.energy[NEW, NEW, :, :]

    # print("V\n", data.coupling_matrix[0, 0, :, :])

    # shift back to surface dependent co-ordinates
    data.qTensor -= data.vib.state_shift[NEW, :, :, NEW]
    # ------------------------------------------------------------------------

    # check that the coupling matrix is symmetric in surfaces
    assert(np.allclose(data.coupling_matrix.transpose(0, 1, 3, 2), data.coupling_matrix))

    (data.coupling_eigvals,
     data.coupling_eigvects
     ) = np.linalg.eigh(data.coupling_matrix, UPLO='L')
    return


def build_numerator(data, vib, outputArray, idx):
    """Calculates the numerator and saves it to the outputArray"""

    # build the M matrix
    np.einsum('abcd, abd, abed->abce',
              data.coupling_eigvects,
              np.exp(-data.tau*data.coupling_eigvals),
              data.coupling_eigvects,
              out=data.M_matrix,
              optimize='optimal'
              )

    # data.numerator = np.broadcast_to(   np.identity(data.states),
    #                                     data.size['BAA']
    #                                     )

    # reset numerator 'storage' to be identity(in the AA dimension)
    data.numerator = np.empty(data.size['BAA'], dtype=F64)
    for b in range(data.block_size):
        data.numerator[b, :, :] = np.identity(data.states)

    for b in range(data.block_size):
        for p in range(data.beads):
            # this is correct
            # data.numerator[b, ...].dot(np.diagflat(vib.omatrix[b, p, :]))
            # this is twice as fast
            # data.numerator[b, ...].dot(np.diag(vib.omatrix[b, p, :]))
            # this is even faster
            data.numerator[b, ...].dot(data.M_matrix[b, p, ...], out=data.numerator[b, :, :])
            data.numerator[b, ...].dot(vib.omatrix[b, p, :, :], out=data.numerator[b, :, :])

    # trace over the surfaces
    outputArray[idx] = np.trace(data.numerator, axis1=1, axis2=2)
    # if np.any(outputArray[idx] < 0.):
    #     log.warning("g(R) had negative values!!!")
    # assert np.all(outputArray[idx] >= 0.), "g(R) must always be positive"
    return


def block_compute_gR(data, result):
    """ Compute and save only g(R) for building data_set to train ML algorithm
     for block_size # of sampled points in each loop"""

    # labels for clarity
    vib = data.vib

    # store results here (these results are not! scaled)
    g = result.scaled_g.view()  # should think about renaming scaled_g?

    for block_index in range(0, data.blocks):

        # indicies
        start = block_index * data.block_size
        end = (block_index + 1) * data.block_size
        sample_view = slice(start, end)

        # generate sample points in collective co-ordinates
        data.draw_sample(sample_view)

        # process the sapmled points
        data.transform_sampled_coordinates(sample_view)

        # build O matricies for system distribution
        build_o_matrix(data, vib.const)

        # compute parts with normal scaling factor
        diagonalize_coupling_matrix(data)
        build_numerator(data, vib.const, g, sample_view)

    result.save_results(end)
    return


def save_gR_with_samples(data, result, input_R_values):
    """ temporary function to save g(R) results with R values
    don't want to pollute BoxResult with extra functions that might not be necessary in the future
    so this will go here for now"""

    result.partial_name = partial(file_name.training_data_g_output().format, P=data.beads, T=data.temperature)
    path = result.compute_path_to_file()

    np.savez(path,
             number_of_samples=result.samples,
             g=result.scaled_g,
             )

    result.partial_name = partial(file_name.training_data_input().format, P=data.beads, T=data.temperature)
    path = result.compute_path_to_file()

    np.savez(path,
             number_of_samples=result.samples,
             input_R_values=input_R_values,
             )
    return


def block_compute_rhoR_from_input_samples(data, result, input_R_values):
    """ Compute and save rho(R) from input R values for testing data_set to train ML algorithm
     for block_size # of sampled points in each loop
    """

    # labels for clarity
    rho = data.rho

    # views
    rho_of_R = result.scaled_rho.view()
    input_R_view = input_R_values.view()

    for block_index in range(0, data.blocks):

        # indicies
        start = block_index * data.block_size
        end = (block_index + 1) * data.block_size
        sample_view = slice(start, end)

        # we copy in the data values that we read in from load_R_samples
        data.qTensor[:] = input_R_view[sample_view, NEW, ...]
        # we transform the dimensionless co-ordinates to surface dependent co-ordinates q = R - d
        data.qTensor -= data.vib.state_shift[NEW, :, :, NEW]

        # build O matricies for system distribution
        build_o_matrix(data, rho.const)
        build_denominator(rho.const, rho_of_R, sample_view)

    # return and let the caller of the function save the results appropriately
    return


def block_compute_gR_from_raw_samples(data, result):
    """ Compute and save g(R) and R values for building data_set to train ML algorithm
     for block_size # of sampled points in each loop
    """

    # labels for clarity
    vib = data.vib

    # store results here (these results are not! scaled)
    g_of_R = result.scaled_g.view()  # should think about renaming scaled_g?
    input_R_values = np.empty(data.size['XNP'], dtype=F64)
    input_view = input_R_values.view()

    for block_index in range(0, data.blocks):

        # indicies
        start = block_index * data.block_size
        end = (block_index + 1) * data.block_size
        sample_view = slice(start, end)

        # generate sample points in surface dependent co-ordinates q = R - d
        # but which were not sampled from
        data.generate_random_R_values(result, input_view, sample_view)

        # build O matricies for system distribution
        build_o_matrix(data, vib.const)

        # compute parts with normal scaling factor
        diagonalize_coupling_matrix(data)
        build_numerator(data, vib.const, g_of_R, sample_view)

    # save results with their sampled co-ordinates
    save_gR_with_samples(data, result, input_R_values)
    return


def block_compute(data, result):
    """Compute the numerator and denominator for block_size # of sampled points in each loop"""

    # block_index_list = [1e1, 1e2, 1e3, 1e4]
    # log.info("Block index list: " + str(block_index_list))

    # labels for clarity
    rho = data.rho
    vib = data.vib

    # store results here
    y_rho = result.scaled_rho.view()
    y_g = result.scaled_g.view()

    # store the combined scaling factor in here
    S12 = np.zeros(data.size['BP'])

    for block_index in range(0, data.blocks):
        # indicies
        start = block_index * data.block_size
        end = (block_index + 1) * data.block_size
        sample_view = slice(start, end)

        # generate sample points in collective co-ordinates
        data.draw_sample(sample_view)

        # process the sapmled points
        data.transform_sampled_coordinates(sample_view)

        # build O matricies for sampling distribution
        build_o_matrix(data, rho.const)
        # build O matricies for system distribution
        build_o_matrix(data, vib.const)

        # compute parts with normal scaling factor
        build_scaling_factors(S12, rho.const, vib.const)
        scale_o_matricies(S12, rho.const, vib.const)

        build_denominator(rho.const, y_rho, sample_view)
        diagonalize_coupling_matrix(data)
        build_numerator(data, vib.const, y_g, sample_view)

    result.save_results(end)

    return


def block_compute_pm(data, result):

    assert isinstance(data, BoxDataPM), "incorrect object type"
    assert isinstance(result, BoxResultPM), "incorrect object type"

    np.set_printoptions(suppress=False)
    # for blockIdx, block in enumerate(range(0, block_size*blocks, block_size)):
    # block_index_list = [1e1, 1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5]
    # log.info("Block index list: " + str(block_index_list))

    # labels for clarity
    rho = data.rho
    vib = data.vib

    # store results here
    y_rho = result.scaled_rho.view()

    y_g = result.scaled_g.view()
    y_gp = result.scaled_gofr_plus.view()
    y_gm = result.scaled_gofr_minus.view()

    # store the combined scaling factor in here
    S12 = np.zeros(data.size['BP'])
    # startTime = time.process_time()
    # log.info("Start: {:f}".format(startTime))
    for block_index in range(0, data.blocks):

        # indicies
        start = block_index * data.block_size
        end = (block_index + 1) * data.block_size
        sample_view = slice(start, end)

        # generate sample points in collective co-ordinates
        data.draw_sample(sample_view)

        # process the sapmled points
        data.transform_sampled_coordinates(sample_view)

        # build O matricies for sampling distribution
        build_o_matrix(data, rho.const)
        # build O matricies for system distribution
        build_o_matrix(data, vib.const)

        # compute parts with normal scaling factor
        build_scaling_factors(S12, rho.const, vib.const)
        scale_o_matricies(S12, rho.const, vib.const)
        build_denominator(rho.const, y_rho, sample_view)
        diagonalize_coupling_matrix(data)
        build_numerator(data, vib.const, y_g, sample_view)

        # Plus
        build_o_matrix(data, vib.const_plus)
        vib.const_plus.omatrix /= S12[..., NEW, NEW]
        build_numerator(data, vib.const_plus, y_gp, sample_view)

        # Minus
        build_o_matrix(data, vib.const_minus)
        vib.const_minus.omatrix /= S12[..., NEW, NEW]
        build_numerator(data, vib.const_minus, y_gm, sample_view)

        # periodically save results to file
        # if (block_index + 1) in block_index_list:
        #     curTime = time.process_time()
        #     timeElapsed = curTime - startTime
        #     startTime = curTime
        #     log.info("Time elapsed: {:f}".format(timeElapsed))
        #     s = "Block index: {:d}\nNumber of samples: {:d}"
        #     log.info(s.format(block_index + 1, end))
        #     result.save_results(end)

    result.save_results(end)
    return


def simple_wrapper(id_data, id_rho=0):
    """Just do simple expval(Z) calculation"""
    np.random.seed(232942)  # pick our seed
    samples = int(1e2)
    Bsize = int(1e2)

    # load the relevant data
    data = BoxData()
    data.id_data = id_data
    data.id_rho = id_rho

    files = file_structure.FileStructure('/work/ngraymon/pimc/', id_data, id_rho)
    data.path_vib_model = files.path_vib_model
    data.path_rho_model = files.path_rho_model

    data.states = 2
    data.modes = 2

    data.samples = samples
    data.beads = 1000
    data.temperature = 300.0
    data.blocks = samples // Bsize
    data.block_size = Bsize

    # setup empty tensors, models, and constants
    data.preprocess()

    # store results here
    results = BoxResult(data=data)
    results.path_root = files.path_rho_results

    block_compute(data, results)
    return


def plus_minus_wrapper(id_data, id_rho=0):
    """Calculate all the possible temp +/- approaches"""
    np.random.seed(232942)  # pick our seed
    samples = int(1e2)
    Bsize = int(1e2)

    delta_beta = constants.delta_beta

    # load the relevant data
    data = BoxDataPM(delta_beta)
    data.id_data = id_data
    # data.id_rho = 0

    files = file_structure.FileStructure('/work/ngraymon/pimc/', id_data, id_rho)
    data.path_vib_model = files.path_vib_model
    data.path_rho_model = files.path_rho_model

    data.states = 3
    data.modes = 6

    data.samples = samples
    data.beads = 20
    data.temperature = 300.00
    data.blocks = samples // Bsize
    data.block_size = Bsize

    # setup empty tensors, models, and constants
    data.preprocess()

    # store results here
    results = BoxResultPM(data=data)
    results.path_root = files.path_rho_results

    # block_compute(data, results)
    block_compute_pm(data, results)
    return


if (__name__ == "__main__"):
    pass
