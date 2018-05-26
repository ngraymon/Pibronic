"""Should hold all server related enums"""

# system imports
from enum import Enum

# local imports

# third party imports


class ServerExecutionParameters(Enum):
    X = "number_of_samples"
    nBlk = "number_of_blocks"
    A = "number_of_states"
    P = "number_of_beads"
    N = "number_of_modes"
    T = "temperature"
    BlkS = "block_size"
    dB = "delta_beta"
    D = "id_data"
    R = "id_rho"
    beta = "beta"
    tau = "tau"
