# timing script to check whicher version of sampling is faster

from .context import pibronic
import multiprocessing as mp
import timeit
import sys
import os

# setup strings
base_setup = (
    "gc.enable()\n"+
    "import numpy as np"
    + "\nnp.random.seed(523536) # pick our seed"
    + "\nfrom numpy import newaxis as NEW"
    + "\nkB = 8.6173303E-5"
    + "\nT = 300.0"
    + "\nbeta = 1. / (kB * T)"
    + "\nA = 3"
    + "\nN = 4"
    + "\nP = 5"
    + "\nX = {samples:d}"
    + "\nO_matrix = np.zeros((P, P))"
    + "\nfor a in range(0, P-1):"
    + "\n\tO_matrix[a, a+1] = 1"
    + "\n\tO_matrix[a+1, a] = 1"
    + "\nO_matrix[0,-1] +=1"
    + "\nO_matrix[-1,0] +=1"
    + "\nO_eigenvals, O_eigenvectors = np.linalg.eigh(O_matrix)"
    + "\nco_tangent_matrix = np.power(np.tanh(np.tile(np.arange(0.025, 0.025*N, 0.025), (A, 1))*1.*(1./(8.6173324E-5*50.0))), -1)"
    + "\nco_secant_matrix = np.power(np.sinh(np.tile(np.arange(0.025, 0.025*N, 0.025), (A, 1))*1.*(1./(8.6173324E-5*50.0))), -1)"
    + "\ncovariance_matrix = np.subtract(2. * co_tangent_matrix[:, :, NEW], co_secant_matrix[:, :, NEW] * O_eigenvals[NEW, NEW, :])"
    )
single_setup = base_setup + (
    "\ninvariance_matrix = np.zeros((X, N, P, P))"
    + "\nmean_P = np.zeros(P)"
    + "\nmean_NP = np.zeros((N,P))"
    + "\nmean_XNP = np.zeros((X,N,P))"
    + "\nunivariate_single = np.empty((X, N, P))"
    + "\nmultivariate_single = np.empty((X, N, P))"
    )
block_setup = base_setup + (
    "\ns_blk = {blk:d}"
    + "\nn_blk = X // s_blk"
    + "\nmean = np.zeros(s_blk)"
    + "\nmean_P = np.zeros((s_blk, P))"
    + "\nmean_NP = np.zeros((s_blk, N, P))"
    + "\nunivariate_blocked = np.empty((X, N, P))"
    + "\nmultivariate_blocked = np.empty((X, N, P))"
    )

# single samples
multi_single = (
    "for x in range(X):"
    + "\n\tfor j in range(N):"
    + "\n\t\tmultivariate_single[x, j, :] ="
    + "np.random.multivariate_normal(mean_P, np.linalg.inv(np.diagflat(covariance_matrix[surface_sample[x], j, :])), size=(1))"
    # + "\nprint('M = ', multivariate_single[0,:,:])"
    )
uni_single = (
    "np.random.seed(523536)\n"
    + "surface_sample = np.random.choice(A, size=(X))\n"
    + "for x in range(X):"
    + "\n\tfor j in range(N):"
    + "\n\t\tfor i in range(P):"
    + "\n\t\t\tunivariate_single[x, j, i] = np.random.normal(loc=0.0, scale=covariance_matrix[surface_sample[x], j, i], size=(1))"
    # + "\nprint('U = ', univariate_single[0:2,:,:])"
    )  
uni_single_P = (
    "np.random.seed(523536)\n"
    + "surface_sample = np.random.choice(A, size=(X))\n"
    + "for x in range(X):"
    + "\n\tfor j in range(N):"
    + "\n\t\tunivariate_single[x, j, :] = np.random.normal(loc=mean_P, scale=covariance_matrix[surface_sample[x], j, :], size=(P))"
    # + "\nprint('U_P = ', univariate_single[0:2,:,:])"
    ) 
uni_single_NP = (
    "np.random.seed(523536)\n"
    + "surface_sample = np.random.choice(A, size=(X))\n"
    + "for x in range(X):"
    + "\n\tunivariate_single[x, :, :] = np.random.normal(loc=mean_NP, scale=covariance_matrix[surface_sample[x], :, :], size=(N,P))"
    # + "\nprint('U_NP = ', univariate_single[0:2,:,:])"
    )
uni_single_XNP = (
    "np.random.seed(523536)\n"
    + "surface_sample = np.random.choice(A, size=(X))\n"
    + "univariate_single[:, :, :] = np.random.normal(loc=mean_XNP, scale=covariance_matrix[surface_sample[:], :, :], size=(X,N,P))"
    # + "\nprint('U_XNP = ', univariate_single[0:2,:,:])"
    )

# block samples
multi_block = (
    "for blk_i in range(1, n_blk+1, 1):"
    + "\n\tnp.random.seed(523536) # pick our seed"
    + "\n\tsurface_sample = np.random.randint(0, A, size=(s_blk))"
    + "\n\tfor j in range(N):"
    + "\n\tmultivariate_blocked[:, j, :] ="
    + "np.random.multivariate_normal(np.zeros(P), np.linalg.inv(np.diagflat(covariance_matrix[surface_sample_array[sample_index], j, :])))"
    # + "\nprint('Mb = ', multivariate_blocked[0:2,:,:])"
    )
uni_block = (
    "np.random.seed(523536)\n"
    + "for b in range(1, n_blk+1, 1):"
    + "\n\tsurface_sample = np.random.choice(A, size=(s_blk))"
    + "\n\tfor j in range(N):"
    + "\n\t\tfor i in range(P):"
    + "\n\t\t\tunivariate_blocked[(b-1)*s_blk:b*s_blk, j, i] = np.random.normal(loc=mean, scale=covariance_matrix[surface_sample[:], j, i], size=(s_blk))"
    # + "\nprint('UB = ', univariate_blocked[0:2,:,:])"
    )    
uni_block_P = (
    "np.random.seed(523536)\n"
    + "for b in range(1, n_blk+1, 1):"
    + "\n\tsurface_sample = np.random.choice(A, size=(s_blk))"
    + "\n\tfor j in range(N):"
    + "\n\t\tunivariate_blocked[(b-1)*s_blk:b*s_blk, j, :] = np.random.normal(loc=mean_P, scale=covariance_matrix[surface_sample[:], j, :], size=(s_blk, P))"
    # + "\nprint('UB_P = ', univariate_blocked[0:2,:,:])"
    )   
uni_block_NP = (
    "np.random.seed(523536)\n"
    + "surface_sample = np.random.choice(A, size=(X))\n"
    + "for b in range(1, n_blk+1, 1):"
    + "\n\tsurface_view = surface_sample[(b-1)*s_blk:b*s_blk].view()"
    + "\n\tunivariate_blocked[(b-1)*s_blk:b*s_blk, :, :] = np.random.normal(loc=mean_NP, scale=covariance_matrix[surface_view[:], :, :], size=(s_blk, N, P))"
    # + "\nprint('UB_NP = ', univariate_blocked[0:2,:,:])"
    )    

# print the sampled values to check that there are no errors
def check_sampling(B, X):
    # timeit.repeat(stmt=uni_single, setup=single_setup.format(samples=X), repeat=1, number=1)
    # timeit.repeat(stmt=uni_single_P, setup=single_setup.format(samples=X), repeat=1, number=1)
    # timeit.repeat(stmt=uni_single_NP, setup=single_setup.format(samples=X), repeat=1, number=1)
    timeit.repeat(stmt=uni_single_XNP, setup=single_setup.format(samples=X), repeat=1, number=1) 
    timeit.repeat(stmt=uni_block_NP, setup=block_setup.format(blk=B, samples=X), repeat=1, number=1)
    # these two don't match
    # timeit.repeat(stmt=uni_block, setup=block_setup.format(blk=B, samples=X), repeat=1, number=1) 
    # timeit.repeat(stmt=uni_block_P, setup=block_setup.format(blk=B, samples=X), repeat=1, number=1)
    sys.exit(0)

def sample(R, N, B, X):
    pstr = "{:<25} speed {}" # :<0.8g
    lambda_list = []
    # sample without blocks
    # lambda_list.extend([lambda:print(pstr.format("multi single", timeit.repeat(stmt=multi_single, setup=single_setup.format(samples=X), repeat=R, number=N)))])
    # lambda_list.extend([lambda:print(pstr.format("uni single", timeit.repeat(stmt=uni_single, setup=single_setup.format(samples=X), repeat=R, number=N)))])
    # lambda_list.extend([lambda:print(pstr.format("uni single_P", timeit.repeat(stmt=uni_single_P, setup=single_setup.format(samples=X), repeat=R, number=N)))])
    # lambda_list.extend([lambda:print(pstr.format("uni single_NP", timeit.repeat(stmt=uni_single_NP, setup=single_setup.format(samples=X), repeat=R, number=N)))])
    lambda_list.extend([lambda:print(pstr.format("uni single_XNP", timeit.repeat(stmt=uni_single_XNP, setup=single_setup.format(samples=X), repeat=R, number=N)))])
    
    # sample with blocks
    # lambda_list.extend([lambda:print(pstr.format("uni block", timeit.repeat(stmt=uni_block, setup=block_setup.format(samples=X, blk=B), repeat=R, number=N)))])
    lambda_list.extend([lambda:print(pstr.format("uni block P", timeit.repeat(stmt=uni_block_P, setup=block_setup.format(samples=X, blk=B), repeat=R, number=N)))])
    lambda_list.extend([lambda:print(pstr.format("uni block NP", timeit.repeat(stmt=uni_block_NP, setup=block_setup.format(samples=X, blk=B), repeat=R, number=N)))])
    
    p = [0]*len(lambda_list)
    for index, func in enumerate(lambda_list):
        p[index] = mp.Process(target=lambda_list[index])
        p[index].start()
    for index, func in enumerate(lambda_list):
        p[index].join()

def main():
    block_size = 1000
    number_of_samples = int(1e5)
    print("Block size: {:}\nNumber of samples: {:}".format(block_size, number_of_samples))
    R = 3
    N = 10
    sample(R, N, block_size, number_of_samples)
    # check_sampling(block_size, number_of_samples)


if __name__ == "__main__":
    main()


