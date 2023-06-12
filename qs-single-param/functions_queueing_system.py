import numpy as np
import math, time


def get_nonzero_indices_p_theta(p_theta_shape, numvals_x, numvals_z, M, K):
    p_indices = np.zeros(p_theta_shape, dtype=bool) #location of 'p' in the transition probability kernel
    theta_not_indices = np.zeros(p_theta_shape, dtype=bool) #location of '1-p' in the transition probability kernel
    for xn in range(numvals_x):
        for zn in range(numvals_z):
            if not (xn==M and zn==0):
                f_x_z = xn- min(zn, xn)
                o_1 = int(f_x_z+1>=K)
                p_indices[f_x_z+1, o_1, xn, zn] = 1
                o_2 = int(f_x_z>=K)
                theta_not_indices[f_x_z, o_2, xn, zn] = 1
    return p_indices, theta_not_indices


def update_pi(pi_curr, p_theta, o_new_ind, numvals_z, dep_rate=None, poisson_arrivals_departures=False, dep_dists=None):
    # if dep_rate is None:
    #     p_theta_reduced = np.sum(p_theta[:, o_new_ind, :, :]*(1/numvals_z), axis=-1).T #numvals(xn), numvals(xn+1) shape
    # else:
    #     p_theta_reduced = np.sum(p_theta[:, o_new_ind, :, :]*np.array([1-dep_rate, dep_rate]).reshape((1,1,-1)), axis=-1).T
    if poisson_arrivals_departures:
        p_theta_reduced = np.sum(p_theta[:, o_new_ind, :, :]*dep_dists[o_new_ind].reshape((1,1,-1)), axis=-1).T
    else:
        p_theta_reduced = np.sum(p_theta[:, o_new_ind, :, :]*np.array([1-dep_rate[o_new_ind], \
                                                                       dep_rate[o_new_ind]]).reshape((1,1,-1)), axis=-1).T
    prod_pi_p_red = pi_curr.reshape((1,-1)) @ p_theta_reduced
    pi_new = prod_pi_p_red / np.sum(prod_pi_p_red)
    # assert len(np.sum(prod_pi_p_red))==1
    pi_new = np.squeeze(pi_new)
    return pi_new

def compute_p_theta(p_theta_old, theta_new, theta_indices, theta_not_indices, M, K):
    """
    Idea: pre-identify indices where p appears and where 1-p appears
    to update the transition kernel, just update the saved indices 
    directly instead of looping over XZ values again (most are zeroes)
    """
    p_theta = np.zeros(p_theta_old.shape)
    eps = 1e-5
    if theta_new<eps:
        theta_new = eps
    elif theta_new>1-eps:
        theta_new=1-eps
    p_theta[theta_indices] = theta_new
    p_theta[theta_not_indices] = 1-theta_new
    p_theta[M,int(M>=K),M,0] = 1
    return p_theta

    
def update_X_O(xk, zk, xikpl1, M, K):
    xtemp = xk-min(zk,xk) + xikpl1
    x_new = min(xtemp, M)
    o_new = (x_new>=K)
    return x_new, o_new


def get_z_o_indices(zk, ok, z_vals, o_vals):
    zind = int(np.argwhere(z_vals==zk))
    oind = int(np.argwhere(o_vals==ok))
    return zind, oind


def compute_ll_next(ll_prev, pi_current, p_theta_current, u_old_ind, o_new_ind):
    p_theta_reduced = p_theta_current[:, o_new_ind, :, u_old_ind].T #shape xn,xnpl1
    prod = pi_current.reshape((-1,1))*p_theta_reduced
    ll_next = ll_prev+np.log(np.sum(prod))
    return ll_next


def compute_log_likelihood_alldata(theta_estimate, O_t, Z_t, T, z_vals, o_vals, x_vals, K, saveLLt=False, dep_rate=None):
    LL = 0
    numvals_x = len(x_vals)
    numvals_z = len(z_vals)
    numvals_o = len(o_vals)
    M = numvals_x-1 #largest value of X
    p_theta = np.zeros((numvals_x, numvals_o, numvals_x, numvals_z))
    theta_indices, theta_not_indices = get_nonzero_indices_p_theta(p_theta.shape, numvals_x, numvals_z, M, K)
    p_theta = compute_p_theta(p_theta, theta_estimate, theta_indices, theta_not_indices, M, K)
    
    pi_t = np.zeros((numvals_x,))
    pi_t[:] = 1/(numvals_x) #start with a uniform belief state (distribution over latent states)
    LL_t = np.zeros((T,))
    for t in range(1,T):
        ztmin1_ind, ot_ind = get_z_o_indices(Z_t[t-1], O_t[t], z_vals, o_vals)
        LL = compute_ll_next(ll_prev=LL, pi_current=pi_t, p_theta_current=p_theta, u_old_ind=ztmin1_ind, o_new_ind=ot_ind)
        pi_t = update_pi(pi_t, p_theta, ot_ind, numvals_z, dep_rate=dep_rate)
        LL_t[t] = LL
    if saveLLt:
        return LL_t
    else:
        return LL


