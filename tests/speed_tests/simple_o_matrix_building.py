# this file is to demonstrate timing for building the coupling matrix
# it serves as a quick confirmation of the speedups acquired from different code


from .context import pibronic
import timeit
import sys
import gc

setupstr = '''
import numpy as np
import scipy.linalg


BOLTZMAN_CONSTANT = 8.6173324E-5 #eV/K
hbar = 1.0

block_size = 10000
number_of_beads = 15
number_of_electronic_surfaces = 8
number_of_normal_modes = 25
temperature = 100.0
beta = 1.0 / (BOLTZMAN_CONSTANT * temperature)
tau = np.float_(beta / number_of_beads)

surface_range = range(number_of_electronic_surfaces)
mode_range    = range(number_of_normal_modes)

omega_array = np.random.random_sample((number_of_electronic_surfaces, number_of_normal_modes))
coth_tensor = np.power(np.tanh(omega_array*hbar*tau), -1)
csch_tensor = np.power(np.sinh(omega_array*hbar*tau), -1)

q_tensor = np.random.random_sample((block_size, number_of_electronic_surfaces, number_of_normal_modes, number_of_beads))

'''

base = '''
o_matrix = np.empty((block_size, number_of_beads, number_of_electronic_surfaces))
q_small = np.empty((2, block_size, number_of_electronic_surfaces, number_of_normal_modes))
alpha = np.empty((2,2, number_of_electronic_surfaces, number_of_normal_modes))
numers_q_alpha_q = np.zeros((block_size, number_of_normal_modes, number_of_beads, number_of_electronic_surfaces))

for bead_i in range(number_of_beads):
    alpha[0,0,:,:] = alpha[1,1,:,:] = coth_tensor[:,:]
    alpha[1,0,:,:] = alpha[0,1,:,:] = -csch_tensor[:,:]
    q_small[0] = q_tensor[:, :, :, bead_i]
    q_small[1] = q_tensor[:, :, :, (bead_i+1)%number_of_beads]
    numers_q_alpha_q[:, :, bead_i, :] = np.einsum('i..., i...', q_small, np.einsum('ij..., i...->j...', alpha, q_small)).swapaxes(1,2)

o_matrix = np.exp(np.multiply(np.float64(-0.5), np.sum(numers_q_alpha_q, axis=1)))
'''

alpha_in_mem = '''
o_matrix = np.empty((block_size, number_of_beads, number_of_electronic_surfaces))
q_small = np.empty((2, block_size, number_of_electronic_surfaces, number_of_normal_modes))
alpha = np.empty((2,2, number_of_electronic_surfaces, number_of_normal_modes))
numers_q_alpha_q = np.zeros((block_size, number_of_normal_modes, number_of_beads, number_of_electronic_surfaces))

alpha[0,0,:,:] = coth_tensor[:,:]
alpha[1,1,:,:] = coth_tensor[:,:]
alpha[1,0,:,:] = -csch_tensor[:,:]
alpha[0,1,:,:] = -csch_tensor[:,:]
for bead_i in range(number_of_beads):
    q_small[0] = q_tensor[:, :, :, bead_i]
    q_small[1] = q_tensor[:, :, :, (bead_i+1)%number_of_beads]
    numers_q_alpha_q[:, :, bead_i, :] = np.einsum('a..., a...', q_small, np.einsum('ab..., ac...->bc...', alpha, q_small)).swapaxes(1,2)

o_matrix = np.exp(np.multiply(np.float64(-0.5), np.sum(numers_q_alpha_q, axis=1)))
'''


print("{:} speed {:}".format("alpha_in_mem", timeit.timeit(stmt=alpha_in_mem, setup=setupstr, number=int(1e1))))
gc.collect()
print("{:} speed {:}".format("base", timeit.timeit(stmt=base, setup=setupstr, number=int(1e1))))