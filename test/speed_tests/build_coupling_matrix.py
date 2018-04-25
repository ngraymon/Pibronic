# this file is to demonstrate timing for building the coupling matrix
# it serves as a quick confirmation of the speedups acquired from different code

from .context import pibronic
import timeit, sys

setupstr = '''
import numpy as np

block_size = 100
number_of_beads = 20
number_of_electronic_surfaces = 8
number_of_normal_modes = 15
tau = 0.5

coupling_operator = np.random.random_sample((number_of_normal_modes, number_of_normal_modes, number_of_electronic_surfaces, number_of_electronic_surfaces))
coupling_matrix = np.random.random_sample((block_size, number_of_beads, number_of_electronic_surfaces, number_of_electronic_surfaces))
q_tensor = np.random.random_sample((block_size, number_of_electronic_surfaces, number_of_normal_modes, number_of_beads))
'''

two_eigs = '''
coupling_matrix = np.einsum('acef, abcdef->afbc', q_tensor, np.einsum('debc, abdf->abcdef', coupling_operator, q_tensor))'''
def eigsum(R,N):
    pstr = "{:} speed \n{:}"
    print(pstr.format("two_eigs", timeit.repeat(stmt=two_eigs, setup=setupstr, repeat=R, number=N)))
    return


comb_eigs = '''
coupling_matrix = np.einsum('acef, debc, abdf->afbc', q_tensor, coupling_operator, q_tensor)'''
def comb_eigsum(R, N):
    eigsum(R, N)
    pstr = "{:} speed \n{:}"
    print(pstr.format("comb_eigs", timeit.repeat(stmt=comb_eigs, setup=setupstr, repeat=R, number=N)))
    return

opt_eigs = '''
coupling_matrix = np.einsum('acef, debc, abdf->afbc', q_tensor, coupling_operator, q_tensor, optimize='optimal')'''
def opt_eigsum(R, N):
    comb_eigsum(R, N)
    pstr = "{:} speed \n{:}"
    print(pstr.format("opt_eigs", timeit.repeat(stmt=opt_eigs, setup=setupstr, repeat=R, number=N)))
    return


def main():
    repeat = 2
    number = int(10)
    opt_eigsum(repeat, number) 


if __name__ == "__main__":
    main()
