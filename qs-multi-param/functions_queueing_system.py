import numpy as np
import math, time


def get_nonzero_indices_state(state_dist_shape, M): #for p(Xn | X_{n-1}, Z_{n-1})
    '''
    indices in prob. distribution p(X_{n+1} | X_n, Z_n) which depend on theta 
    '''
    theta_indices = np.zeros(state_dist_shape, dtype=bool) #location of 'p' in the transition probability kernel
    theta_not_indices = np.zeros(state_dist_shape, dtype=bool) #location of '1-p' in the transition probability kernel
    numvals_x = state_dist_shape[0]
    numvals_z = state_dist_shape[2]
    for xn in range(numvals_x):
        for zn in range(numvals_z):
            if not ((xn==M and zn==0) or (xn==0 and zn==1)):
                f_x_z = xn- min(zn, xn)
                theta_indices[f_x_z+1, xn, zn] = 1
                theta_not_indices[f_x_z, xn, zn] = 1
    return theta_indices, theta_not_indices

def get_nonzero_indices_obs(obs_dist_shape, K, unsymmetric=False): #for p(On | Xn)
    '''
    indices in p(O_k|X_k) which depend on theta(2) and theta(3) (if it exists- when unsymmetric is True)
    '''
    phi_indices = np.zeros(obs_dist_shape, dtype=bool)
    phi_not_indices = np.zeros(obs_dist_shape, dtype=bool)
    
    zeta_indices = np.zeros(obs_dist_shape, dtype=bool)
    zeta_not_indices = np.zeros(obs_dist_shape, dtype=bool)
    
    numvals_o = obs_dist_shape[0]
    numvals_x = obs_dist_shape[1]
    for xn in range(numvals_x):
        if xn>=K:
            phi_indices[0, xn] = 1
            phi_not_indices[1, xn] = 1
        else: #xn<K
            if not unsymmetric:
                phi_not_indices[0, xn] = 1
                phi_indices[1, xn] = 1
            else:
                zeta_not_indices[0, xn] = 1
                zeta_indices[1, xn] = 1
                
    if not unsymmetric:
        return phi_indices, phi_not_indices
    else:
        return [(phi_indices, zeta_indices), (phi_not_indices, zeta_not_indices)]


def get_full_dist(dist_list):
    '''
    takes a list of two distributions p(X_{n+1}|X_n, Z_n) and p(O_{n+1}|X_{n+1}) and returns p(X_{n+1}, O_{n_1}|X_n, Z_n)
    '''
    state_dist, obs_dist = dist_list[0], dist_list[1]
    numvals_o = obs_dist.shape[0]
    numvals_x = obs_dist.shape[1]
    numvals_z = state_dist.shape[2]
    dist = np.zeros((numvals_x, numvals_o, numvals_x, numvals_z))
    for on in range(numvals_o):
        dist[:,on,:,:] = state_dist[:,:,:]*(obs_dist[on,:].reshape((-1,1,1)))
    return dist


def update_dist(theta_bar_new, dist_shape_list, theta_bar_limits, theta_bar_indices, M, unsymmetric=False):
    """
    dist_shape_list: 2 entries: state_dist_shape (tuple), obs_dist_shape (tuple)
    theta_bar_new: list with 2 entries: theta and phi
    theta_bar_indices: list with 2 (or 3) entries: (theta_indices, theta_not_indices), (phi_indices, phi_not_indices), (zeta_indices, zeta_not_indices)
    theta_bar_limits: [(theta_min, theta_max), (phi_min, phi_max)]
    
    Idea: pre-identify indices where p appears and where 1-p appears
    to update the transition kernel, just update the saved indices 
    directly instead of looping over XZ values again (most are zeroes)
    
    
    """
    state_dist_new = np.zeros(dist_shape_list[0])
    obs_dist_new = np.zeros(dist_shape_list[1])
    dist_list_new = [state_dist_new, obs_dist_new]
    thr = 1e-5 #threshold
    for i in range(len(theta_bar_new)): #restrict to corresponding limits
        theta_bar_new[i] = min( max(theta_bar_new[i], theta_bar_limits[i][0]+thr), theta_bar_limits[i][1]-thr)
    
    dist_list_new[0][theta_bar_indices[0][0]] = theta_bar_new[0] #theta
    dist_list_new[0][theta_bar_indices[0][1]] = 1-theta_bar_new[0] #theta_not (1-theta)
    dist_list_new[0][M, M, 0] = 1 #at terminal state
    dist_list_new[0][0, 0, 1] = 1 #at 0-state, with departure=1, Xnew = 0-1+arr (will be 0 irrespective of arr) 
    
    dist_list_new[1][theta_bar_indices[1][0]] = theta_bar_new[1] #phi
    dist_list_new[1][theta_bar_indices[1][1]] = 1-theta_bar_new[1] #phi_not (1-phi)
    
    if unsymmetric: #another parameter
        dist_list_new[1][theta_bar_indices[2][0]] = theta_bar_new[2] #zeta
        dist_list_new[1][theta_bar_indices[2][1]] = 1-theta_bar_new[2] #zeta_not (1-zeta)
        
    
    return dist_list_new


def update_pi_fixedparam(pi_curr, dist_list, o_new_ind, numvals_z, dep_rate): #old, without using perturbation in pi
    """
    update the nonlinear filter when the parameter is fixed
    """
    p_theta = get_full_dist(dist_list)
    p_theta_reduced = np.sum(p_theta[:, o_new_ind, :, :]*np.array([1-dep_rate[o_new_ind], \
                                                                       dep_rate[o_new_ind]]).reshape((1,1,-1)), axis=-1).T
    prod_pi_p_red = pi_curr.reshape((1,-1)) @ p_theta_reduced
    pi_new = prod_pi_p_red / np.sum(prod_pi_p_red)
    pi_new = np.squeeze(pi_new)
    return pi_new

def update_pi_new(pi_curr, o_new_ind, dep_rate, omega_bar_t, theta_bar_t, gamma_bar, dist_shape_list, \
             theta_bar_limits, theta_bar_indices, M, unsymmetric): #old, without using perturbation in pi
    """
    update the nonlinear filter with perturbation
    """
    pert_vector = np.sin(omega_bar_t)
    theta_bar_pert = theta_bar_t + gamma_bar*pert_vector #elementwise product
    dist_pert_list = update_dist(theta_bar_pert, dist_shape_list, theta_bar_limits, theta_bar_indices, M, unsymmetric)
    p_th_pert = get_full_dist(dist_pert_list)
    p_theta_reduced = np.sum(p_th_pert[:, o_new_ind, :, :]*np.array([1-dep_rate[o_new_ind], \
                                                                       dep_rate[o_new_ind]]).reshape((1,1,-1)), axis=-1).T
    prod_pi_p_red = pi_curr.reshape((1,-1)) @ p_theta_reduced
    pi_new = prod_pi_p_red / np.sum(prod_pi_p_red)
    pi_new = np.squeeze(pi_new)
    return pi_new


def update_X_O(xk, zk, xikpl1, M, K, theta_bar_star, unsymmetric=False): #stochastic observations now
    xtemp = xk-min(zk,xk) + xikpl1
    x_new = min(xtemp, M)
    # o_new = (x_new>=K)
    if x_new>=K:
        o_new = np.random.binomial(n=1, p=1-theta_bar_star[1]) #phi
    else:
        if unsymmetric:
            o_new = np.random.binomial(n=1, p=theta_bar_star[2]) #zeta
        else:
            o_new = np.random.binomial(n=1, p=theta_bar_star[1])
    return x_new, o_new


def get_z_o_indices(zk, ok, z_vals, o_vals):
    zind = int(np.argwhere(z_vals==zk))
    oind = int(np.argwhere(o_vals==ok))
    return zind, oind


def compute_ll_next(ll_prev, pi_current, dist_list_current, dep_rate, o_new_ind):
    p_theta_current = get_full_dist(dist_list_current)
    # p_theta_reduced = p_theta_current[:, o_new_ind, :, u_old_ind].T #shape xn,xnpl1
    p_theta_reduced = np.sum(p_theta_current[:, o_new_ind, :, :]*np.array([1-dep_rate[o_new_ind], \
                                                                       dep_rate[o_new_ind]]).reshape((1,1,-1)), axis=-1).T
    prod = pi_current.reshape((-1,1))*p_theta_reduced
    ll_next = ll_prev+np.log(np.sum(prod))
    return ll_next


def compute_log_likelihood_alldata(theta_bar_estimate, O_t, T, z_vals, o_vals, x_vals, K, theta_bar_limits, \
                                   saveLLt=False, dep_rate=None, unsymmetric=False):
    LL = 0
    numvals_x = len(x_vals)
    numvals_z = len(z_vals)
    numvals_o = len(o_vals)
    M = numvals_x-1 #largest value of X
    # p_theta = np.zeros((numvals_x, numvals_o, numvals_x, numvals_z))
    # theta_indices, theta_not_indices = get_nonzero_indices_p_theta(p_theta.shape, numvals_x, numvals_z, M, K)
    # p_theta = compute_p_theta(p_theta, theta_estimate, theta_indices, theta_not_indices, M, K)
    state_dist_shape = (numvals_x, numvals_x, numvals_z) #p(X_{n+1} | X_n, Z_n)
    obs_dist_shape = (numvals_o, numvals_x) #p(O_{n+1} | X_{n+1})
    dist_shape_list = [state_dist_shape, obs_dist_shape]
    theta_indices, theta_not_indices = get_nonzero_indices_state(state_dist_shape, M)
    indices_obs_params = get_nonzero_indices_obs(obs_dist_shape, K, unsymmetric)
    if not unsymmetric:
        phi_indices, phi_not_indices = indices_obs_params[0], indices_obs_params[1]
        theta_bar_indices = [(theta_indices, theta_not_indices), (phi_indices, phi_not_indices)]
    else:
        phi_indices, phi_not_indices = indices_obs_params[0][0], indices_obs_params[1][0]
        zeta_indices, zeta_not_indices = indices_obs_params[0][1], indices_obs_params[1][1]
        theta_bar_indices = [(theta_indices, theta_not_indices), (phi_indices, phi_not_indices), (zeta_indices, zeta_not_indices)]
    dist_list = update_dist(theta_bar_estimate, dist_shape_list, theta_bar_limits, theta_bar_indices, M, unsymmetric)
    
    pi_t = np.zeros((numvals_x,))
    pi_t[:] = 1/(numvals_x) #start with a uniform belief state (distribution over latent states)
    LL_t = np.zeros((T,))
    for t in range(1,T):
        ot_ind = int(np.argwhere(o_vals==O_t[t]))
        LL = compute_ll_next(ll_prev=LL, pi_current=pi_t, dist_list_current=dist_list, dep_rate=dep_rate, o_new_ind=ot_ind)
        pi_t = update_pi_fixedparam(pi_t, dist_list, ot_ind, numvals_z, dep_rate=dep_rate) #when computing the log likelihood, just need normal pi upadte (no perturbation)
        LL_t[t] = LL
    if saveLLt:
        return LL_t
    else:
        return LL


