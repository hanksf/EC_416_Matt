import numpy as np
import numpy.polynomial.chebyshev as chebyshev
import numba
from stationary_functions import stationary

def backward_iterate_linear(kplus,Pi,Phi,beta,alpha,delta,z,k):
    """
    Uses backwards iteration to update the capital policy function
    """
    k_endogenous = (1/Phi)*(1+Phi*k-beta*((1-delta)+Pi@(alpha*z[:,np.newaxis]*k**(alpha-1)+Phi*(kplus-k))))

    k_new = np.empty_like(kplus)
    for zi in range(len(z)):
        k_new[zi, :] = np.interp(k,k_endogenous[zi,:],k)
    return k_new

def ss_policy_linear(alpha,beta,delta, Phi,Pi,z,klow,khigh, N,tol = 1E-9):
    k = np.linspace(klow,khigh,num=N,endpoint=True)
    kplus = np.broadcast_to(k, (len(z), len(k)))
    for it in range(1000):
        kplus = backward_iterate_linear(kplus,Pi,Phi,beta,alpha,delta,z,k)
        if it % 10 == 1 and  np.max(np.abs(kplus - kold)) < tol:
            print(f'convergence in {it} iterations!')
            return kplus
        kold = kplus

def chebyshev_nodes(xlow, xhigh, N):
    standard_nodes = -np.cos((2*np.arange(1, N+1)-1)/(2*N)*np.pi)
    x = xlow + (xhigh-xlow)/2*(1+standard_nodes)
    return x

def chebyshev_interp(x, y):
    X = chebyshev.chebvander(x, len(x)-1)
    return np.linalg.solve(X, y)

def backward_iterate_chebyshev(kplus,Pi,Phi,beta,alpha,delta,z,k):
    """
    Uses backwards iteration to update the capital policy function
    """
    k_endogenous = (1/Phi)*(1+Phi*k-beta*((1-delta)+Pi@(alpha*z[:,np.newaxis]*k**(alpha-1)+Phi*(kplus-k))))

    q = np.empty_like(kplus)
    k_new = np.empty_like(kplus)
    for zi in range(len(z)):
        q = chebyshev_interp(k_endogenous[zi,:], k)
        k_new[zi, :] = chebyshev.chebval(k, q)
    return k_new


def ss_policy_chebyshev(alpha,beta,delta, Phi,Pi,z,klow,khigh, N):
    k = chebyshev_nodes(klow,khigh, N)
    kplus = np.broadcast_to(k, (len(z), len(k)))
    for it in range(1000):
        kplus = backward_iterate_chebyshev(kplus,Pi,Phi,beta,alpha,delta,z,k)
        if it % 10 == 1 and  np.max(np.abs(kplus - kold)) < 1E-10:
            print(f'convergence in {it} iterations!')
            return kplus
        kold = kplus

def ss_policy_chebyshev_remapped(alpha,beta,delta, Phi,Pi,z,klow,khigh, N):
    k = chebyshev_nodes(klow,khigh, N)
    kplus = np.broadcast_to(k, (len(z), len(k)))
    for it in range(1000):
        kplus = backward_iterate_chebyshev_remapped(kplus,Pi,Phi,beta,alpha,delta,z,k,klow,khigh)
        if it % 10 == 1 and  np.max(np.abs(kplus - kold)) < 1E-10:
            print(f'convergence in {it} iterations!')
            return kplus
        kold = kplus

def euler_equation_errors_linear(kplus,alpha,beta,delta, Phi,Pi,z,k,klow,khigh, N):
    k_grid = np.linspace(klow,khigh,num=N,endpoint=True)
    kplus_dense = np.empty((len(z),len(k)))
    for i in range(len(z)):
        kplus_dense[i,:] = np.interp(k,k_grid,kplus[i,:])
    LHS = 1+Phi*(kplus_dense-k)
    RHS = np.empty_like(LHS)
    k2 = np.empty_like(kplus_dense)
    for i, zi in enumerate(z):
        for j in range(len(z)):
            k2[j,:] = np.interp(kplus_dense[i,:],k,kplus_dense[j,:])
        RHS[i,:]=beta*((1-delta)+Pi[i,:]@(alpha*z[:,np.newaxis]*kplus_dense[i,:]**(alpha-1)+Phi*(k2-kplus_dense[i,:])))
    return np.log10(np.abs(LHS-RHS))

def euler_equation_errors(kplus,alpha,beta,delta, Phi,Pi,z,klow,khigh, N):
    k = np.linspace(klow,khigh,num=N,endpoint=True)
    LHS = 1+Phi*(kplus-k)
    RHS = np.empty_like(LHS)
    k2 = np.empty_like(kplus)
    for i, zi in enumerate(z):
        for j in range(len(z)):
            k2[j,:] = np.interp(kplus[i,:],k,kplus[j,:])
        RHS[i,:]=beta*((1-delta)+Pi[i,:]@(alpha*z[:,np.newaxis]*kplus[i,:]**(alpha-1)+Phi*(k2-kplus[i,:])))
    return np.log10(np.abs(LHS-RHS))

def ee_error_chebyshev(q, k,alpha,beta,delta, Phi,Pi,z):
    kplus = chebyshev.chebval(k, q)
    LHS = 1+Phi*(kplus-k)
    RHS = np.empty_like(LHS)
    for i, zi in enumerate(z):
        k2 = chebyshev.chebval(kplus[i,:], q)
        c_inner = chebyshev.chebval(kplus[i, :], q.T)
        RHS[i,:]=beta*((1-delta)+Pi[i,:]@(alpha*z[:,np.newaxis]*kplus[i,:]**(alpha-1)+Phi*(k2-kplus[i,:])))
    return np.log10(np.abs(LHS-RHS))


@numba.njit
def interpolate_coord(x, xq):
    """Get representation xqi, xqpi of xq interpolated against x:
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points

    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]
    xqi = np.empty(nxq, dtype=np.int64)
    xqpi = np.empty(nx)

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi
    return xqi, xqpi

@numba.njit
def forward_iterate(D, Pi_T, kplus_i, kplus_pi):
    # first step: update using endogenous capital policy
    # initialize Dnew with zeros because we're looping through D, not Dnew
    Dnew = np.zeros_like(D)
    for zi in range(D.shape[0]):
        for ki in range(D.shape[1]):
            i = kplus_i[zi, ki]
            pi = kplus_pi[zi, ki]
            d = D[zi, ki]
            
            Dnew[zi, i] += d*pi
            Dnew[zi, i+1] += d*(1-pi)
    
    # second step: update using transpose of Markov matrix for exogenous state z
    # take Pi_T itself as input for memory efficiency
    return Pi_T @ Dnew

def ergodic_dist(Pi, kplus_i, kplus_pi):
    # start by getting stationary distribution of z
    pi = stationary(Pi)
    
    # need to initialize joint distribution of (z, k), assume uniform on k
    nK = kplus_i.shape[1]
    D = np.outer(pi, np.full(nK, 1/nK))
    
    # Pi.T is a "view" on Pi with the wrong memory order, copy this to get right memory order (faster operations)
    Pi_T = Pi.T.copy()
    
    # now iterate forward until convergence
    for it in range(100_000):
        Dnew = forward_iterate(D, Pi_T, kplus_i, kplus_pi)
        
        # only check convergence every 20 iterations for efficiency
        if it % 20 == 0 and np.max(np.abs(Dnew-D)) < 1E-10:
            print(f'Convergence after {it} forward iterations!')
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    
    return D

import numba
@numba.njit
def chebval_matt(x, q):
    b_2 = 0.
    b_1 = 0.
    for k in range(len(q)-1, 0, -1):
        b = q[k] + 2*x*b_1 - b_2
        b_2 = b_1
        b_1 = b
    return q[0] + x*b_1 - b_2