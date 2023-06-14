import numpy as np
from functions_queueing_system import *

def update_omega(omega, b_t, eps=0.1):
    return omega*(1)+eps*b_t #linear increase instead of exponential


def update_theta_es(theta_old, p_theta_old, a_t, omega_t, pi_t, ot_ind, numvals_z, theta_indices, \
                    theta_not_indices, M, K, gamma=0.1, dep_rate=None):
    """
    Includes a departure rate (prob of departure), and momentum
    """
    p_th_pert = compute_p_theta(p_theta_old, theta_old+gamma*np.sin(omega_t), theta_indices, theta_not_indices, M, K)
    p_th_pert_xn = np.sum(np.sum(p_th_pert[:, ot_ind, :, :], axis=0)*np.array([1-dep_rate[ot_ind],\
                                                                           dep_rate[ot_ind]]).reshape((1,-1)),axis=1)
    p_th_pert_cond = np.sum(pi_t.reshape((-1,))*p_th_pert_xn.reshape((-1,))) #p(O_{t}\mid O^{t-1})
    theta_new = theta_old + a_t*(1/gamma)*np.sin(omega_t)*np.log(p_th_pert_cond)
    
    thr = 1e-5 #threshold- to limit theta to (thr, 1-thr)
    if theta_new<thr:
        theta_new = thr
    elif theta_new>1-thr:
        theta_new=1-thr
    return theta_new


def estimate_theta_es(theta_init, theta_star, M = 20, K = 3, T = 2000, \
                      alpha=0.8, beta=0.7, eps=1, gamma=0.1,\
                      dep_rate=[0.1, 0.7], A=0.1, B=10):
    
    numvals_x = M+1
    numvals_o = 2 #binary
    numvals_z = 2
    X_qlen = np.zeros((T,))
    O_obs = np.zeros((T,))
    xi_arr = np.zeros((T,))
    Z_dep = np.zeros((T,))
        
    #initialize
    X_qlen[0] = np.random.randint(numvals_x)
    O_obs[0] = (X_qlen[0]>=K)
    Z_dep[0] = np.random.randint(numvals_z)
    xi_arr[0] = np.random.binomial(1,theta_star)

    # store all values of X, O, Z in arrays to easily retrieve the index later on
    x_vals = np.arange(numvals_x) #all values of x
    o_vals = np.arange(numvals_o)
    z_vals = np.arange(numvals_z)

    pi_t = np.zeros((numvals_x,))
    pi_t[:] = 1/(numvals_x) #start with a uniform belief state (distribution over latent states)
    pi_t_all = np.zeros((numvals_x, T))
    pi_t_all[:,0] = pi_t[:]

    theta_hat_t = np.zeros((T,)) #estimate through time
    
    ######################################## INITIAL ESTIMATE OF THETA
    theta_hat_t[0]= theta_init # a random initial guess of the unknown parameter

    p_theta = np.zeros((numvals_x, numvals_o, numvals_x, numvals_z))
    theta_indices, theta_not_indices = get_nonzero_indices_p_theta(p_theta.shape, numvals_x, numvals_z, M, K)
    p_theta = compute_p_theta(p_theta, theta_hat_t[0], theta_indices, theta_not_indices, M, K)

    omega_t = 0.1 #initial value
    omega_t_all = np.zeros((T,))
    omega_t_all[0]=omega_t
 
    for t in range(1,T):
        #dynamics of the process
        xi_arr[t] = np.random.binomial(1, theta_star)
        X_qlen[t], O_obs[t] = update_X_O(X_qlen[t-1], Z_dep[t-1], xi_arr[t], M, K)
        Z_dep[t] = (1-O_obs[t])*np.random.binomial(1, p=dep_rate[0]) + (O_obs[t])*np.random.binomial(1, p=dep_rate[1])

        ztmin1_ind, ot_ind = get_z_o_indices(Z_dep[t-1], O_obs[t], z_vals, o_vals)
        tred = math.ceil(t/100)
        a_t = A/tred**alpha
        b_t = B/tred**beta

        omega_t_all[t]=update_omega(omega_t_all[t-1], b_t, eps)
        
        theta_hat_t[t] = update_theta_es(theta_hat_t[t-1], p_theta, a_t, omega_t_all[t-1], pi_t_all[:,t-1], \
                                         ot_ind, numvals_z, theta_indices, theta_not_indices, M, K, gamma, \
                                         dep_rate=dep_rate) 
        
        pi_t_all[:,t] = update_pi_new(pi_t_all[:,t-1], theta_hat_t[t-1], gamma, omega_t_all[t-1], theta_indices, \
                                      theta_not_indices, M, K, p_theta, ot_ind, dep_rate)
        p_theta = compute_p_theta(p_theta, theta_hat_t[t], theta_indices, theta_not_indices, M, K)
        
    results = {'X':X_qlen, 'O':O_obs, 'xi':xi_arr, 'Z':Z_dep, 'pi_t_all':pi_t_all, 'theta_hat_t':theta_hat_t,\
               'omega_t':omega_t_all, 'p_theta':p_theta}
    return results