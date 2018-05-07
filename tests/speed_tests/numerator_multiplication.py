# this file is to demonstrate timing for building the coupling matrix
# it serves as a quick confirmation of the speedups acquired from different code

from .context import pibronic
import timeit
import sys

setupstr = '''
import numpy as np
import scipy.linalg

import os
os.sched_setaffinity(0, range(16))

block_size = 10000
number_of_beads = 200
number_of_electronic_surfaces = 8

numerator   = np.zeros((block_size))
numerator_1 = np.random.random_sample((block_size, number_of_electronic_surfaces, number_of_electronic_surfaces))
M_matrix    = np.random.random_sample((block_size, number_of_beads, number_of_electronic_surfaces, number_of_electronic_surfaces))
o_matrix    = np.empty((block_size, number_of_beads, number_of_electronic_surfaces, number_of_electronic_surfaces))
temp_om_matrix = np.random.random_sample((block_size, number_of_beads, number_of_electronic_surfaces, number_of_electronic_surfaces))

for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        o_matrix[sample_index, bead_index, :, :] = np.diagflat(np.random.random_sample(number_of_electronic_surfaces))
'''


# compare matmul with dot 
all_numpy = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        temp_om_matrix[sample_index, bead_index, :, :] = np.dot(o_matrix[sample_index, bead_index, :, :], M_matrix[sample_index, bead_index, :, :])'''
matmul = '''
for sample_index in range(block_size):
    temp_om_matrix[sample_index, :, :, :] = np.matmul(o_matrix[sample_index, :, :, :], M_matrix[sample_index, :, :, :])'''
full_vectorized_matmul = '''
temp_om_matrix = np.matmul(o_matrix, M_matrix)'''

def matmul_vs_dot(R, N):
    pstr = "{:} speed \n{:}"
    print(pstr.format("all_numpy", timeit.repeat(stmt=all_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("matmul", timeit.repeat(stmt=matmul, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("full_vectorized_matmul", timeit.repeat(stmt=full_vectorized_matmul, setup=setupstr, repeat=R, number=N)))

# compare multi_dot with multiple dots
multiple_dots = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        numerator_1[sample_index, :, :] = np.dot(numerator_1[sample_index, :, :], np.dot(o_matrix[sample_index, bead_index, :, :], M_matrix[sample_index, bead_index, :, :]))'''

multi_dot = '''
temp_om_matrix = np.matmul(o_matrix, M_matrix)
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        numerator_1[sample_index, :, :] = np.dot(numerator_1[sample_index, :, :], temp_om_matrix[sample_index, bead_index, :, :])'''

multi_mat = '''
temp_om_matrix = np.matmul(o_matrix, M_matrix)
for bead_index in range(number_of_beads):
    numerator_1 = np.matmul(numerator_1, temp_om_matrix[:, bead_index, :, :])'''

einsum_mat = '''
temp_om_matrix = np.matmul(o_matrix, M_matrix)
for bead_index in range(number_of_beads):
    numerator = np.einsum('acd, acd-> a', numerator_1, temp_om_matrix[:, bead_index, :, :])'''

einsum_full = '''
temp_om_matrix = np.matmul(o_matrix, M_matrix)
for bead_index in range(number_of_beads):
    numerator = np.einsum('acd, abcd, abcd-> a', numerator_1, o_matrix[:, bead_index, :, :], M_matrix[:, bead_index, :, :])'''

def multi_dot_vs_multiple_dots(R, N):
    pstr = "{:} speed \n{:}"
    print(pstr.format("multiple dots", timeit.repeat(stmt=multiple_dots, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("multi_dot", timeit.repeat(stmt=multi_dot, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("multi_mat", timeit.repeat(stmt=multi_mat, setup=setupstr, repeat=R, number=N)))
    #print(pstr.format("einsum_mat", timeit.repeat(stmt=einsum_mat, setup=setupstr, repeat=R, number=N)))
    #print(pstr.format("einsum_full", timeit.repeat(stmt=einsum_full, setup=setupstr, repeat=R, number=N)))


def main():
    R = 2
    N = 10
    #matmul_vs_dot(R, N)
    multi_dot_vs_multiple_dots(R, N)


if __name__ == "__main__":
    main()


