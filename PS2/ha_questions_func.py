import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy import interpolate
import stationary_functions as st


def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return np.sum(pi * (x - np.sum(pi * x)) ** 2)



def fast_rouwenhorst_income(rho, sigma_eps, N=7):
    """
    Faster function to generate Rouwenhorst approximation to an AR(1)
    """
    # parametrize Rouwenhorst Markov matrix for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    Pi = st.fast_recursion(N,p,Pi)
        
    pi = st.stationary(Pi)

    # analytically determined spacing of points should give exactly right sd
    s = np.linspace(-1, 1, N)
    s *= (sigma_eps / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))
    
    return y, pi, Pi


def backward_iterate(cplus, up, up_inv, beta, Pi, r, y, a):
    # step one: get consumption on endogenous gridpoints 
    cendog = up_inv(beta * (1+r) * Pi @ up(cplus))
    
    # step two: solve for consumption on regular gridpoints implied by Euler equation
    coh = y[:, np.newaxis] + (1+r)*a
    c = np.empty_like(cendog)
    for s, y in enumerate(y):
        c[s, :] = np.interp(coh[s, :], cendog[s, :] + a, cendog[s, :])
    
    # step three: enforce a_+ >= amin, assuming amin is lowest gridpoint of a
    aplus = coh - c
    aplus[aplus < a[0]] = a[0]
    c = coh - aplus
    
    return c, aplus


def agrid(amax,N,amin=0,pivot=0.1):
    """Grid with a+pivot evenly log-spaced between amin+pivot and amax+pivot"""
    a = np.geomspace(amin+pivot,amax+pivot,N) - pivot
    a[0] = amin # make sure *exactly* equal to amin
    return a

def ss_policy(up, up_inv, beta, Pi, r, y, a, tol = 1E-9):
    # guess initial value for consumption function as 10% of cash on hand
    coh = y[:, np.newaxis] + (1+r)*a
    
    #What type of guess performs best?
    #Matt suggested convergence is faster for those who are more myopic, i.e near the constraint
    #Therefore you might what to guess something that is close to the behaviour you expect
    #from those away from the constraint
    c = 0.1*coh
    
    # iterate until convergence
    for it in range(2000):
        c, aplus = backward_iterate(c, up, up_inv, beta, Pi, r, y, a)
        
        if it % 10 == 1 and np.max(np.abs(c - cold)) < tol:
            return c, aplus
        
        cold = c        

def ss_policy_plot(up, up_inv, beta, Pi, r, y, a, tol = 1E-9):
    # guess initial value for consumption function as 10% of cash on hand
    coh = y[:, np.newaxis] + (1+r)*a
    
    #What type of guess performs best?
    #Matt suggested convergence is faster for those who are more myopic, i.e near the constraint
    #Therefore you might what to guess something that is close to the behaviour you expect
    #from those away from the constraint
    c = 0.1*coh
    
    # iterate until convergence
    for it in range(2000):
        c, aplus = backward_iterate(c, up, up_inv, beta, Pi, r, y, a)
        if it % 40 == 0 and it != 0:
            plt.plot(a,np.log(np.max(np.abs(c - cold),axis=0)),label=f'iter {it}')
        if it % 10 == 1 and np.max(np.abs(c - cold)) < tol:
            plt.legend()
            plt.show()
            return c, aplus
        
        cold = c        

def ss_policy_relativeplot(up, up_inv, beta, Pi, r, y, a, tol = 1E-9):
    # guess initial value for consumption function as 10% of cash on hand
    coh = y[:, np.newaxis] + (1+r)*a
    
    #What type of guess performs best?
    #Matt suggested convergence is faster for those who are more myopic, i.e near the constraint
    #Therefore you might what to guess something that is close to the behaviour you expect
    #from those away from the constraint
    c = 0.1*coh
    
    # iterate until convergence
    for it in range(2000):
        c, aplus = backward_iterate(c, up, up_inv, beta, Pi, r, y, a)
        if it % 40 == 0 and it != 0:
            plt.plot(a,np.log(np.max(np.abs((c - cold)/cold),axis=0)),label=f'iter {it}')
        if it % 10 == 1 and np.max(np.abs(c - cold)) < tol:
            plt.legend()
            plt.show()
            return c, aplus
        
        cold = c        

def euler_equation_errors(c_pol,a_pol,up, beta, Pi, r, y, a):
    """
    function to calculate the euler equation errors in the HA model
    Disregards when the constraint is binding
    """
    LHS = up(c_pol)
    RHS = np.empty_like(c_pol)
    for i, yi in enumerate(y):
        c_inner = np.empty_like(c_pol)
        for j in range(len(y)):
            c_inner[j,:] = np.interp(a_pol[i,:],a,c_pol[j,:])
        RHS[i,:] = beta*(1+r)*Pi[i,:]@up(c_inner)
    errors = np.log10(np.abs(LHS-RHS))
    errors[a_pol==a[0]]=0
    return errors

def backward_iterate_cubic(cplus, up, up_inv, beta, Pi, r, y, a):
    # step one: get consumption on endogenous gridpoints 
    cendog = up_inv(beta * (1+r) * Pi @ up(cplus))
    
    # step two: solve for consumption on regular gridpoints implied by Euler equation
    coh = y[:, np.newaxis] + (1+r)*a
    c = np.empty_like(cendog)
    for s, y in enumerate(y):
        tck = interpolate.splrep(cendog[s, :] + a, cendog[s, :])
        c[s, :] = interpolate.splev(coh[s, :], tck)
    
    # step three: enforce a_+ >= amin, assuming amin is lowest gridpoint of a
    aplus = coh - c
    aplus[aplus < a[0]] = a[0]
    c = coh - aplus
    
    return c, aplus



def ss_policy_cubic(up, up_inv, beta, Pi, r, y, a, tol = 1E-9):
    # guess initial value for consumption function as 10% of cash on hand
    coh = y[:, np.newaxis] + (1+r)*a
    
    #What type of guess performs best?
    #Matt suggested convergence is faster for those who are more myopic, i.e near the constraint
    #Therefore you might what to guess something that is close to the behaviour you expect
    #from those away from the constraint
    c = 0.1*coh
    
    # iterate until convergence
    for it in range(2000):
        c, aplus = backward_iterate_cubic(c, up, up_inv, beta, Pi, r, y, a)
        
        if it % 10 == 1 and np.max(np.abs(c - cold)) < tol:
            return c, aplus
        
        cold = c        

def euler_equation_errors_cubic(c_pol,a_pol,up, beta, Pi, r, y, a):
    """
    function to calculate the euler equation errors in the HA model
    Disregards when the constraint is binding
    """
    LHS = up(c_pol)
    RHS = np.empty_like(c_pol)
    for i, yi in enumerate(y):
        c_inner = np.empty_like(c_pol)
        for j in range(len(y)):
            tck = interpolate.splrep(a, c_pol[j, :])
            c_inner[j, :] =  interpolate.splev(a_pol[i,:], tck)
        RHS[i,:] = beta*(1+r)*Pi[i,:]@up(c_inner)
    errors = np.log10(np.abs(LHS-RHS))
    errors[a_pol==a[0]]=0
    return errors