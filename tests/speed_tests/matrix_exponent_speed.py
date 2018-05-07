# this file is to demonstrate timing for building the coupling matrix
# it serves as a quick confirmation of the speedups acquired from different code

from .context import pibronic
import timeit
import sys

setupstr = '''
import numpy as np
import scipy.linalg
#import os
#print(os.sched_getaffinity(0))
#os.sched_setaffinity(0, range(16))
#print(os.sched_getaffinity(0))

block_size = 100
number_of_beads = 20
number_of_electronic_surfaces = 8
tau = 0.5

coupling_matrix = np.random.random_sample((block_size, number_of_beads, number_of_electronic_surfaces, number_of_electronic_surfaces))
eigvect = np.empty((block_size, number_of_beads, number_of_electronic_surfaces, number_of_electronic_surfaces))
eigval  = np.empty((block_size, number_of_beads, number_of_electronic_surfaces))
M_matrix = np.empty((block_size, number_of_beads, number_of_electronic_surfaces, number_of_electronic_surfaces))
'''

super_vectorized_numpy = '''
eigval[:, :, :], eigvect[:, :, :, :] = np.linalg.eigh(coupling_matrix[:, :, :, :])
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect))'''
extra_vectorized_numpy = '''
for sample_index in range(block_size):
        eigval[sample_index, :, :], eigvect[sample_index, :, :, :] = np.linalg.eigh(coupling_matrix[sample_index, :, :, :])
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect))'''
all_numpy = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        eigval[sample_index, bead_index, :], eigvect[sample_index, bead_index, :, :] = np.linalg.eigh(coupling_matrix[sample_index, bead_index, :, :])
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect))'''
def compare_vectorized_decomposition(R, N):
    pstr = "{:} speed \n{:}"
    print(pstr.format("super_vectorized_numpy", timeit.repeat(stmt=super_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("extra_vectorized_numpy", timeit.repeat(stmt=extra_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("all_numpy", timeit.repeat(stmt=all_numpy, setup=setupstr, repeat=R, number=N)))
    return

opt_super_vectorized_numpy = '''
eigval[:, :, :], eigvect[:, :, :, :] = np.linalg.eigh(coupling_matrix[:, :, :, :])
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect, optimize='optimal'), optimize='optimal')'''
opt_extra_vectorized_numpy = '''
for sample_index in range(block_size):
        eigval[sample_index, :, :], eigvect[sample_index, :, :, :] = np.linalg.eigh(coupling_matrix[sample_index, :, :, :])
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect, optimize='optimal'), optimize='optimal')'''
opt_all_numpy = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        eigval[sample_index, bead_index, :], eigvect[sample_index, bead_index, :, :] = np.linalg.eigh(coupling_matrix[sample_index, bead_index, :, :])
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect, optimize='optimal'), optimize='optimal')'''
def compare_opt_vectorized_decomposition(R, N):
    compare_vectorized_decomposition(R, N)
    pstr = "{:} speed \n{:}"
    print(pstr.format("opt_super_vectorized_numpy", timeit.repeat(stmt=opt_super_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("opt_extra_vectorized_numpy", timeit.repeat(stmt=opt_extra_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("opt_all_numpy", timeit.repeat(stmt=opt_all_numpy, setup=setupstr, repeat=R, number=N)))
    return


single_super_vectorized_numpy = '''
eigval[:, :, :], eigvect[:, :, :, :] = np.linalg.eigh(coupling_matrix[:, :, :, :])
M_matrix = np.einsum('abcd, abc, abcd->abcd', eigvect, np.exp(-tau*eigval), eigvect)'''
single_extra_vectorized_numpy = '''
for sample_index in range(block_size):
        eigval[sample_index, :, :], eigvect[sample_index, :, :, :] = np.linalg.eigh(coupling_matrix[sample_index, :, :, :])
M_matrix = np.einsum('abcd, abc, abcd->abcd', eigvect, np.exp(-tau*eigval), eigvect)'''
single_all_numpy = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        eigval[sample_index, bead_index, :], eigvect[sample_index, bead_index, :, :] = np.linalg.eigh(coupling_matrix[sample_index, bead_index, :, :])
M_matrix = np.einsum('abcd, abc, abcd->abcd', eigvect, np.exp(-tau*eigval), eigvect)'''
def compare_comb_vectorized_decomposition(R, N):
    compare_vectorized_decomposition(R, N)
    pstr = "{:} speed \n{:}"
    print(pstr.format("single_super_vectorized_numpy", timeit.repeat(stmt=single_super_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("single_extra_vectorized_numpy", timeit.repeat(stmt=single_extra_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("single_all_numpy", timeit.repeat(stmt=single_all_numpy, setup=setupstr, repeat=R, number=N)))
    return



scipy_diagonalization_no_checks = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        eigval[sample_index, bead_index, :], eigvect[sample_index, bead_index, :, :] = scipy.linalg.eigh(coupling_matrix[sample_index, bead_index, :, :], turbo=True, check_finite=False)
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect))'''
scipy_diagonalization = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        eigval[sample_index, bead_index, :], eigvect[sample_index, bead_index, :, :] = scipy.linalg.eigh(coupling_matrix[sample_index, bead_index, :, :])
M_matrix = np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect))'''
scipy_matrix_exponentiation = '''
coupling_matrix *= -tau
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        M_matrix[sample_index, bead_index, :, :] = scipy.linalg.expm(coupling_matrix[sample_index, bead_index, :, :])'''
slow_scipy_matrix_exponentiation = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        M_matrix[sample_index, bead_index, :, :] = scipy.linalg.expm(-tau*coupling_matrix[sample_index, bead_index, :, :])'''
def compare_with_scipy(R, N):
    compare_vectorized_decomposition(R, N)
    pstr = "{:} speed \n{:}"
    print(pstr.format("scipy_diagonalization_no_checks", timeit.repeat(stmt=scipy_diagonalization_no_checks, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("scipy_diagonalization", timeit.repeat(stmt=scipy_diagonalization, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("scipy_matrix_exponentiation", timeit.repeat(stmt=scipy_matrix_exponentiation, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("slow_scipy_matrix_exponentiation", timeit.repeat(stmt=slow_scipy_matrix_exponentiation, setup=setupstr, repeat=R, number=N)))
    return



# clearly using the out parameter doesn't give significant speed ups
# super_vectorized_numpy speed      62.33340008370578
# extra_vectorized_numpy speed      65.7367700394243
# all_numpy speed                   114.97570619359612
# out_super_vectorized_numpy speed  62.0371239669621
# out_extra_vectorized_numpy speed  76.97209451533854
# out_all_numpy speed               147.663432540372
out_super_vectorized_numpy = '''
eigval[:, :, :], eigvect[:, :, :, :] = np.linalg.eigh(coupling_matrix[:, :, :, :])
np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect), out=M_matrix)'''
out_extra_vectorized_numpy = '''
for sample_index in range(block_size):
        eigval[sample_index, :, :], eigvect[sample_index, :, :, :] = np.linalg.eigh(coupling_matrix[sample_index, :, :, :])
np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect), out=M_matrix)'''
out_all_numpy = '''
for sample_index in range(block_size):
    for bead_index in range(number_of_beads):
        eigval[sample_index, bead_index, :], eigvect[sample_index, bead_index, :, :] = np.linalg.eigh(coupling_matrix[sample_index, bead_index, :, :])
np.einsum('abcd, abcd->abcd', eigvect, np.einsum('abc, abcd->abcd', np.exp(-tau*eigval), eigvect), out=M_matrix)'''
def compare_out_paramater(R, N):
    compare_vectorized_decomposition(R, N)
    pstr = "{:} speed \n{:}"
    print(pstr.format("out_super_vectorized_numpy", timeit.repeat(stmt=out_super_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("out_extra_vectorized_numpy", timeit.repeat(stmt=out_extra_vectorized_numpy, setup=setupstr, repeat=R, number=N)))
    print(pstr.format("out_all_numpy",  timeit.repeat(stmt=out_all_numpy, setup=setupstr, repeat=R, number=N)))
    return


def main():

    repeat = 4
    number = int(1e3)

    #compare_out_paramater(repeat, number)
    #compare_opt_vectorized_decomposition(repeat, number)
    compare_comb_vectorized_decomposition(repeat, number) # optimal for now
    #compare_vectorized_decomposition(repeat, number)
    #compare_with_scipy(repeat, number)


if __name__ == "__main__":
    main()





