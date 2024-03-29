import sys

import numpy as np
import scipy.spatial.distance

import numpy as np
from numba import njit
import scipy.spatial

def mmdagg(
    seed, X, Y, alpha, kernel_type, approx_type, weights_type, l_minus, l_plus, B1, B2, B3
):
    """
    Compute MMDAgg as defined in Algorithm 1 in our paper using the collection of
    bandwidths defined in Eq. (16) and the weighting strategies proposed in Section 5.1.
    inputs: seed: integer random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            weights_type: "uniform", "decreasing", "increasing" or "centred" (Section 5.1 of our paper)
            l_minus: integer (for collection of bandwidths Eq. (16) in our paper)
            l_plus: integer (for collection of bandwidths Eq. (16) in our paper)
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
    output: result of MMDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    if approx_type == "wild_bootstrap":
        approx_type = "wild bootstrap"
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    assert X.shape[1] == Y.shape[1]
    assert 0 < alpha  and alpha < 1
    assert kernel_type in ["gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2", "all_l1", "all_l2", "all", "all_plus_g", "laplace_gaussian"]
    assert approx_type in ["wild bootstrap"] # "permutation" to be added
    assert weights_type in ["uniform", "decreasing", "increasing", "centred"]
    assert l_plus >= l_minus

    # define l
    if "l1" in kernel_type or kernel_type == "laplace":
        l = "l1"
    elif "l2" in kernel_type:
        l = "l2"
    elif kernel_type in ["gaussian", "imq"]:
        l = "l2sq"
    elif kernel_type == "all":
        l = "all"

    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)
    
    # define bandwidth_multipliers and weights
    scale = 1
    bandwidths = np.array([2 ** (i / scale) * median_bandwidth for i in range(l_minus * scale, l_plus * scale + 1)])
    
    # create weights
    N =  len(bandwidths)
    if kernel_type == "laplace_gaussian":
        N = 2 * N
    if kernel_type == "all":
        n_kernels = 3 * 2
        N = n_kernels * N
    if kernel_type == "all_plus_g":
        n_kernels = 3 * 2
        N_old = N
        N = (n_kernels + 1) * N
        N_old = n_kernels * N_old
    elif kernel_type in ["all_l1", "all_l2"]:
        n_kernels = 3 
        N = n_kernels * N
    weights = create_weights(N, weights_type)

    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in our paper)
    M  = np.zeros((N, B1 + B2 + 1))  
    rs = np.random.RandomState(seed)
    if approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+B2+1) 
        if kernel_type in ["all_l1", "all_l2"]:
            l = kernel_type[-2:]
            pairwise_matrix = compute_pairwise_matrix(X, Y, l)
            kernels = ["matern_" + str(j) + ".5_" + l for j in range(n_kernels)]
            for j in range(n_kernels):
                kernel = kernels[j]
                for i in range(int(N / n_kernels)):
                    bandwidth = bandwidths[i]
                    # compute kernel matrix for bandwidth
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    # set diagonal elements to zero
                    mutate_K(K, approx_type)
                    # compute MMD bootstrapped values
                    M[int(N / n_kernels) * j + i] = np.sum(R * (K @ R), 0)
        elif kernel_type == "all":
            pairwise_matrix_l1 = compute_pairwise_matrix(X, Y, "l1")
            pairwise_matrix_l2 = compute_pairwise_matrix(X, Y, "l2")
            kernels = ["matern_" + str(j) + ".5_" + ll for j in range(int(n_kernels / 2)) for ll in ("l1", "l2")]
            for j in range(n_kernels):
                kernel = kernels[j]
                l = kernel[-2:]
                pairwise_matrix = pairwise_matrix_l1 if l == "l1" else pairwise_matrix_l2
                for i in range(int(N / n_kernels)):
                    bandwidth = bandwidths[i]
                    # compute kernel matrix for bandwidth
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    # set diagonal elements to zero
                    mutate_K(K, approx_type)
                    # compute MMD bootstrapped values
                    M[int(N / n_kernels) * j + i] = np.sum(R * (K @ R), 0)
        elif kernel_type == "all_plus_g":
            pairwise_matrix_l1 = compute_pairwise_matrix(X, Y, "l1")
            pairwise_matrix_l2 = compute_pairwise_matrix(X, Y, "l2")
            kernels = ["matern_" + str(j) + ".5_" + ll for j in range(int(n_kernels / 2)) for ll in ("l1", "l2")]
            for j in range(n_kernels):
                kernel = kernels[j]
                l = kernel[-2:]
                pairwise_matrix = pairwise_matrix_l1 if l == "l1" else pairwise_matrix_l2
                for i in range(int(N_old / n_kernels)):
                    bandwidth = bandwidths[i]
                    # compute kernel matrix for bandwidth
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    # set diagonal elements to zero
                    mutate_K(K, approx_type)
                    # compute MMD bootstrapped values
                    M[int(N_old / n_kernels) * j + i] = np.sum(R * (K @ R), 0)
            pairwise_matrix = pairwise_matrix_l2 
            for i in range(int(N_old / n_kernels)):
                bandwidth = bandwidths[i]
                # compute kernel matrix for bandwidth
                K = kernel_matrix(pairwise_matrix, l, "gaussian", bandwidth)
                # set diagonal elements to zero
                mutate_K(K, approx_type)
                # compute MMD bootstrapped values
                M[N_old + i] = np.sum(R * (K @ R), 0)
        elif kernel_type == "laplace_gaussian":
            l = "l1"
            pairwise_matrix = compute_pairwise_matrix(X, Y, l)
            for i in range(int(N / 2)):
                bandwidth = bandwidths[i]
                # compute kernel matrix for bandwidth
                K = kernel_matrix(pairwise_matrix, l, "laplace", bandwidth)
                # set diagonal elements to zero
                mutate_K(K, approx_type)
                # compute MMD bootstrapped values
                M[i] = np.sum(R * (K @ R), 0)
            l = "l2sq"
            pairwise_matrix = compute_pairwise_matrix(X, Y, l)
            for i in range(int(N / 2)):
                bandwidth = bandwidths[i]
                # compute kernel matrix for bandwidth
                K = kernel_matrix(pairwise_matrix, l, "gaussian", bandwidth)
                # set diagonal elements to zero
                mutate_K(K, approx_type)
                # compute MMD bootstrapped values
                M[int(N / 2) + i] = np.sum(R * (K @ R), 0)
        else:
            pairwise_matrix = compute_pairwise_matrix(X, Y, l)
            for i in range(N):
                bandwidth = bandwidths[i]
                # compute kernel matrix for bandwidth
                K = kernel_matrix(pairwise_matrix, l, kernel_type, bandwidth)
                # set diagonal elements to zero
                mutate_K(K, approx_type)
                # compute MMD bootstrapped values
                M[i] = np.sum(R * (K @ R), 0)
    else:
        raise ValueError(
            'The value of approx_type should be "wild bootstrap".'
        )
    MMD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1 + 1])  # (N, B1+1)
    M2 = M[:, B1 + 1:]  # (N, B2)

    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0
    u_max = np.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for i in range(N):
            quantiles[i] = M1_sorted[
                i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
            ]
        P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min
        
    # Step 3: output test result
    reject = False
    for i in range(N):
        if ( MMD_original[i] 
            > M1_sorted[i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1]
        ):
            reject = True
    if reject:
        return 1
    return 0 


@njit
def mutate_K(K, approx_type):
    """
    Mutate the kernel matrix K depending on the type of approximation.
    inputs: K: kernel matrix of size (m+n,m+n) consisting of 
               four matrices of sizes (m,m), (m,n), (n,m) and (n,n)
               m and n are the numbers of samples from p and q respectively
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            
    output: if approx_type is "permutation" then the estimate is MMD_a (Eq. (3)) and 
               the matrix K is mutated to have zero diagonal entries
            if approx_type is "wild bootstrap" then the estimate is MMD_b (Eq. (6)),
               we have m = n and the matrix K is mutated so that the four matrices 
               have zero diagonal entries
    """
    if approx_type == "permutation":
        for i in range(K.shape[0]):
            K[i, i] = 0      
    if approx_type == "wild bootstrap":
        m = int(K.shape[0] / 2)  # m = n
        for i in range(m):
            K[i, i] = 0
            K[m + i, m + i] = 0
            K[i, m + i] = 0 
            K[m + i, i] = 0

def compute_pairwise_matrix(X, Y, l):
    """
    Compute the pairwise distance matrix between all the points in X and Y,
    in L1 norm or L2 norm or L2 norm squared.

    inputs: X: (m,d) array of samples
            Y: (m,d) array of samples
            l: "l1" or "l2" or "l2sq"
    output: (2m,2m) pairwise distance matrix
    """
    Z = np.concatenate((X, Y))
    if l == "l1":
        return scipy.spatial.distance.cdist(Z, Z, 'cityblock')
    elif l == "l2":
        return scipy.spatial.distance.cdist(Z, Z, 'euclidean')
    elif l == "l2sq":
        return scipy.spatial.distance.cdist(Z, Z, 'sqeuclidean')
    else:
        raise ValueError("Third input should either be 'l1', 'l2' or 'l2sq'.")

def kernel_matrix(pairwise_matrix, l, kernel_type, bandwidth):
    """
    Compute kernel matrix for a given kernel_type and bandwidth. 

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel_type: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel_type must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel_type == "gaussian" and l == "l2sq":
        return  np.exp(-d / bandwidth)
    if kernel_type == "gaussian" and l == "l2":
        return  np.exp(-d ** 2)
    elif kernel_type == "imq" and l == "l2sq":
        return (1 + d / bandwidth) ** (-0.5)
    elif (kernel_type == "matern_0.5_l1" and l == "l1") or (kernel_type == "matern_0.5_l2" and l == "l2") or (kernel_type == "laplace" and l == "l1"):
        return  np.exp(-d)
    elif (kernel_type == "matern_1.5_l1" and l == "l1") or (kernel_type == "matern_1.5_l2" and l == "l2"):
        return (1 + np.sqrt(3) * d) * np.exp(- np.sqrt(3) * d)
    elif (kernel_type == "matern_2.5_l1" and l == "l1") or (kernel_type == "matern_2.5_l2" and l == "l2"):
        return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(- np.sqrt(5) * d)
    elif (kernel_type == "matern_3.5_l1" and l == "l1") or (kernel_type == "matern_3.5_l2" and l == "l2"):
        return (1 + np.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * np.sqrt(7) / 3 / 5 * d ** 3) * np.exp(- np.sqrt(7) * d)
    elif (kernel_type == "matern_4.5_l1" and l == "l1") or (kernel_type == "matern_4.5_l2" and l == "l2"):
        return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * np.exp(- 3 * d)
    else:
        raise ValueError(
            'The values of l and kernel_type are not valid.'
        )


def compute_median_bandwidth_subset(seed, X, Y, max_samples=500):
    """
    Compute the median L^2-distance between all the points in X and Y using at
    most max_samples samples.
    inputs: seed: non-negative integer
            X: (m,d) array of samples
            Y: (m,d) array of samples
            max_samples: number of samples used to compute the median (int or None)
    output: median bandwidth (float)
    """
    assert X.shape[1] == Y.shape[1]
    Z = np.concatenate((X[:max_samples], Y[:max_samples]))
    median = np.median(scipy.spatial.distance.pdist(Z, "euclidean"))
    return median


def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of our paper.
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = np.array([1 / N,] * N)
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = np.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights
