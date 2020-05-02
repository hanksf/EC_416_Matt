import numpy as np
import numba

@numba.njit
def stationary_via_eigenvalues(Pi):
    """
    Function to calculate stationary distribution of Markov chain with transition matrix Pi
    using the observation that the stationary distribution will be an eigenvector of the
    transpose with eigenvalue 1
    """
    eigenvalues, eigenvectors = np.linalg.eig(Pi.T)
    index = np.argmax(eigenvalues)
    return eigenvectors[:,index]/np.sum(eigenvectors[:,index])


@numba.njit
def fast_recursion(N,p,Pi_initial):
    """
    Function containing the recursion in the Rouwenhorst method. Complied using numba for speed.
    Breaks down the vectorised components of the original function to increase speed.
    """
    Pi = Pi_initial
    #recursion that builds from n=3 to n=N
    for n in range(3,N+1):
        Pi_old = Pi
        Pi = np.zeros((n,n))
        for i in range(n-1):
            for j in range(n-1):
                Pi[i, j] += p * Pi_old[i, j]
                Pi[i, j+1] += (1 - p) * Pi_old[i, j]
                Pi[i+1, j] += (1 - p) * Pi_old[i, j]
                Pi[i+1, j+1] += p * Pi_old[i, j]
        for i in range(n-2):
            for j in range(n):
                Pi[i+1, j] /= 2
    return Pi

def fast_rouwenhorst(rho, sigma_eps, N=7):
    """
    Faster function to generate Rouwenhorst approximation to an AR(1)
    """
    # parametrize Rouwenhorst Markov matrix for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    Pi = fast_recursion(N,p,Pi)
        
    pi = stationary(Pi)

    # analytically determined spacing of points should give exactly right sd
    psi = sigma_eps / np.sqrt(1 - rho**2) * np.sqrt(N - 1)
    s = np.linspace(-psi, psi, N)

    return s, pi, Pi


@numba.njit
def stationary(Pi, tol=1E-11, maxit=10_000):
    """Find stationary distribution of a Markov chain by iteration."""
    pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    for it in range(maxit):
        pi_new = pi @ Pi
        #if you expect to do a lot of iterations then it is cheaper to only check 
        #every 10 times rather than every time
        #the check here is really inefficient because you want to break as soon
        #as you find any value >tol
        if it % 10 == 0 and np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError('No convergence to stationary distribution!')

    # normalize before returning to purge slight numerical error
    return pi_new / pi_new.sum()