# this file is to demonstrate timing for building the coupling matrix
# it serves as a quick confirmation of the speedups acquired from different code

from .context import pibronic
import multiprocessing as mp
import timeit
import sys
import os

setupstr = '''gc.enable()
import numpy as np
import scipy.linalg


BOLTZMAN_CONSTANT = 8.6173324E-5 #eV/K
hbar = 1.0

ASSERTION_VALUE = 113130.692352

number_of_samples = {sample:d}
block_size = {blk:d}
number_of_beads = 200
number_of_electronic_surfaces = 8
number_of_normal_modes = 15
temperature = 100.0
beta = 1.0 / (BOLTZMAN_CONSTANT * temperature)
tau = np.float_(beta / number_of_beads)

surface_range = range(number_of_electronic_surfaces)
mode_range    = range(number_of_normal_modes)

#omega_array = np.random.random_sample((number_of_electronic_surfaces, number_of_normal_modes))
omega_array = np.empty((number_of_electronic_surfaces, number_of_normal_modes))
omega_array.fill(0.2)
coth_tensor = np.power(np.tanh(omega_array*hbar*tau), -1)
csch_tensor = np.power(np.sinh(omega_array*hbar*tau), -1)

#q_tensor = np.random.random_sample((block_size, number_of_electronic_surfaces, number_of_normal_modes, number_of_beads))
q_tensor = np.empty((block_size, number_of_electronic_surfaces, number_of_normal_modes, number_of_beads))
q_tensor.fill(1.5)

o_matricies = np.zeros((number_of_samples, number_of_beads, number_of_electronic_surfaces))

q_small = np.empty((2, block_size, number_of_electronic_surfaces, number_of_normal_modes))
alpha = np.empty((2,2, number_of_electronic_surfaces, number_of_normal_modes))

A = np.empty((block_size, number_of_normal_modes, number_of_beads, number_of_electronic_surfaces))
B = np.empty((block_size, number_of_normal_modes, number_of_beads, number_of_electronic_surfaces))

test = np.empty((block_size, number_of_normal_modes, number_of_beads, number_of_electronic_surfaces))

'''
dot =  '''
q_small = np.empty(2)
numers_q_alpha_q = np.zeros((block_size, number_of_normal_modes, number_of_beads, number_of_electronic_surfaces))
for block_index in range(1, (number_of_samples // block_size)+1, 1):
    OMATRIX = o_matricies.view()[(block_index-1)*block_size:block_index*block_size,:,:]

    alpha[0,0,:,:] = alpha[1,1,:,:] = coth_tensor[:,:].view()
    alpha[1,0,:,:] = alpha[0,1,:,:] = -csch_tensor[:,:].view()

    #Q1 = q_tensor.view()
    #Q2 = np.roll(q_tensor.view(), shift=-1, axis=3)

    for sample in range(block_size):
        for surf in range(number_of_electronic_surfaces):
            for mode in range(number_of_normal_modes):   
                for bead in range(number_of_beads):
                    q_small[0] = q_tensor[sample, surf, mode, bead].view()
                    q_small[1] = q_tensor[sample, surf, mode, (bead+1)%number_of_beads].view()
                    numers_q_alpha_q[sample, mode, bead, surf] = np.dot(q_small, np.dot(alpha[:,:,surf,mode], q_small))

    OMATRIX[:] = np.exp(np.multiply(np.float64(-0.5), np.sum(numers_q_alpha_q, axis=1)))
assert(np.isclose(np.sum(np.trace(o_matricies, axis1=1, axis2=2)),ASSERTION_VALUE))'''
base = '''
numers_q_alpha_q = np.zeros((block_size, number_of_electronic_surfaces, number_of_beads, number_of_normal_modes))
for block_index in range(1, (number_of_samples // block_size)+1, 1):
    OMATRIX = o_matricies.view()[(block_index-1)*block_size:block_index*block_size,:,:]
    for bead_i in range(number_of_beads):
        alpha[0,0,:,:] = alpha[1,1,:,:] = coth_tensor[:,:].view()
        alpha[1,0,:,:] = alpha[0,1,:,:] = -csch_tensor[:,:].view()
        q_small[0] = q_tensor[:, :, :, bead_i].view()
        q_small[1] = q_tensor[:, :, :, (bead_i+1)%number_of_beads].view()
        numers_q_alpha_q[:, :, bead_i, :] = np.einsum('i..., ij..., j...->...', q_small, alpha, q_small)
    OMATRIX[:] = np.exp(np.multiply(np.float64(-0.5), np.sum(numers_q_alpha_q, axis=3))).swapaxes(1,2)
#print(np.sum(np.trace(o_matricies, axis1=1, axis2=2)))
assert(np.isclose(np.sum(np.trace(o_matricies, axis1=1, axis2=2)),ASSERTION_VALUE))'''
alpha_in_mem = '''
numers_q_alpha_q = np.zeros((block_size, number_of_beads, number_of_electronic_surfaces, number_of_normal_modes))
for block_index in range(1, (number_of_samples // block_size)+1, 1):
    OMATRIX = o_matricies.view()[(block_index-1)*block_size:block_index*block_size,:,:]
    alpha[0,0,:,:] = alpha[1,1,:,:] = coth_tensor[:,:]
    alpha[1,0,:,:] = alpha[0,1,:,:] = -csch_tensor[:,:]
    for bead_i in range(number_of_beads):
        q_small[0] = q_tensor[:, :, :, bead_i]
        q_small[1] = q_tensor[:, :, :, (bead_i+1)%number_of_beads]
        numers_q_alpha_q[:, bead_i, :, :] = np.einsum('i..., ij..., j...->...', q_small, alpha, q_small)
    OMATRIX[:] = np.exp(np.multiply(np.float64(-0.5), np.sum(numers_q_alpha_q, axis=3)))
assert(np.isclose(np.sum(np.trace(o_matricies, axis1=1, axis2=2)),ASSERTION_VALUE))'''
alpha_in_mem_view = '''
numers_q_alpha_q = np.zeros((block_size, number_of_beads, number_of_electronic_surfaces, number_of_normal_modes))
for block_index in range(1, (number_of_samples // block_size)+1, 1):
    OMATRIX = o_matricies.view()[(block_index-1)*block_size:block_index*block_size,:,:]
    alpha[0,0,:,:] = alpha[1,1,:,:] = coth_tensor[:,:].view()
    alpha[1,0,:,:] = alpha[0,1,:,:] = -csch_tensor[:,:].view()
    for bead_i in range(number_of_beads):
        q_small[0] = q_tensor[:, :, :, bead_i].view()
        q_small[1] = q_tensor[:, :, :, (bead_i+1)%number_of_beads].view()
        numers_q_alpha_q[:, bead_i, :, :] = np.einsum('i..., ij..., j...->...', q_small, alpha, q_small)
    OMATRIX[:] = np.exp(np.multiply(np.float64(-0.5), np.sum(numers_q_alpha_q, axis=3)))
assert(np.isclose(np.sum(np.trace(o_matricies, axis1=1, axis2=2)),ASSERTION_VALUE))'''
numerical_method = '''
for block_index in range(1, (number_of_samples // block_size)+1, 1):
    OMATRIX = o_matricies.view()[(block_index-1)*block_size:block_index*block_size,:,:]
    Q1 = q_tensor.view()
    Q2 = np.roll(q_tensor.view(), shift=-1, axis=3)
    OMATRIX[:] = np.sum((coth_tensor[np.newaxis, :, :, np.newaxis] * (Q1**2. + Q2**2.)) -  2.*csch_tensor[np.newaxis, :, :, np.newaxis] * Q1 * Q2 , axis=2).swapaxes(1,2)
    OMATRIX[:] *= -0.5
    OMATRIX[:] = np.exp(OMATRIX[:])
#print(np.sum(np.trace(o_matricies, axis1=1, axis2=2)))
assert(np.isclose(np.sum(np.trace(o_matricies, axis1=1, axis2=2)),ASSERTION_VALUE))'''


def o_matrix(R, N, block_size, number_of_samples):
    pstr = "{:<25} speed {}" # :<0.8g
    lambda_list = []
    lambda_list.extend([lambda:print(pstr.format("numerical_method", sorted(timeit.repeat(stmt=numerical_method, setup=setupstr.format(blk=block_size, sample=number_of_samples), repeat=R, number=N), key=float)))])
    lambda_list.extend([lambda:print(pstr.format("alpha_in_mem", sorted(timeit.repeat(stmt=alpha_in_mem, setup=setupstr.format(blk=block_size, sample=number_of_samples), repeat=R, number=N), key=float)))])
    lambda_list.extend([lambda:print(pstr.format("alpha_in_mem_view", sorted(timeit.repeat(stmt=alpha_in_mem_view, setup=setupstr.format(blk=block_size, sample=number_of_samples), repeat=R, number=N), key=float)))])
    lambda_list.extend([lambda:print(pstr.format("base", sorted(timeit.repeat(stmt=base, setup=setupstr.format(blk=block_size, sample=number_of_samples), repeat=R, number=N), key=float)))])
    #lambda_list.extend([lambda:print(pstr.format("dot", timeit.repeat(stmt=dot, setup=setupstr, repeat=R, number=N)))])
    
    p = [0]*len(lambda_list)
    for index, func in enumerate(lambda_list):
        p[index] = mp.Process(target=lambda_list[index])
        p[index].start()
    for index, func in enumerate(lambda_list):
        p[index].join()

def main():
    block_size = 100
    number_of_samples = int(1e5)
    print("Block size: {:}\nNumber of samples: {:}".format(block_size, number_of_samples))
    R = 3
    N = 10
    o_matrix(R, N, block_size, number_of_samples)


if __name__ == "__main__":
    main()




