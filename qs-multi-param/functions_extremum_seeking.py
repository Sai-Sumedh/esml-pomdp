import numpy as np
from functions_queueing_system import *

def update_omega(omega_bar, b_t, w0_bar):
    return omega_bar+b_t*w0_bar #linear increase instead of exponential


def update_theta_es(theta_bar_old, dist_shape_list, a_t, omega_bar_t, pi_t, ot_ind, numvals_z, theta_bar_indices, dep_rate,\
                    theta_bar_limits, M, K, gamma_bar=np.array([0.1, 0.1]), unsymmetric=False):
    """
    Includes a departure rate (prob of departure)
    """
    pert_vector = np.sin(omega_bar_t)
    theta_bar_pert = theta_bar_old + gamma_bar*pert_vector #elementwise product
    dist_pert_list = update_dist(theta_bar_pert, dist_shape_list, theta_bar_limits, theta_bar_indices, M, unsymmetric)
    p_th_pert = get_full_dist(dist_pert_list)
        
    p_th_pert_xn = np.sum(np.sum(p_th_pert[:, ot_ind, :, :], axis=0)*np.array([1-dep_rate[ot_ind],\
                                                                               dep_rate[ot_ind]]).reshape((1,-1)),axis=1)
    p_th_pert_cond = np.sum(pi_t.reshape((-1,))*p_th_pert_xn.reshape((-1,))) #p(O_{t}\mid O^{t-1})
    theta_bar_new = theta_bar_old + (a_t/gamma_bar)*pert_vector*np.log(p_th_pert_cond)
    thr = 1e-5 #threshold- to limit theta to (thr, 1-thr)

    for i in range(len(theta_bar_new)): #restrict to corresponding limits
        theta_bar_new[i] = min( max(theta_bar_new[i], theta_bar_limits[i][0]+thr), theta_bar_limits[i][1]-thr)
    return theta_bar_new


def estimate_theta_es(theta_bar_init, theta_bar_star, theta_bar_dim=2, M = 100, K = 50, T = 10000,\
                      alpha=0.8, beta=0.7, w0_bar=[1,1/2], gamma_bar=[0.1,0.1], theta_bar_limits=[[0, 1],[0, 0.5]], tau=100, \
                      dep_rate=[0.1, 0.7], A=0.01, B=10):
    '''
    run the ESML algorithm and return results
    '''
    if theta_bar_dim==2:
        unsymmetric = False
    elif theta_bar_dim==3:
        unsymmetric = True
    else:
        raise ValueError("Allowed values for theta_bar_dim are {2,3}")
    
    numvals_x = M+1
    numvals_o = 2 #binary
    numvals_z = 2
    
    X_qlen = np.zeros((T,))
    O_obs = np.zeros((T,))
    xi_arr = np.zeros((T,))
    Z_dep = np.zeros((T,))
    
    #initialize
    X_qlen[0] = np.random.randint(numvals_x)
    O_obs[0] = (X_qlen[0]>=K) #just one point so can initialize any way
    Z_dep[0] = np.random.randint(numvals_z)
    xi_arr[0] = np.random.binomial(1,theta_bar_star[0])

    # store all values of X, O, Z in arrays to easily retrieve the index later on
    x_vals = np.arange(numvals_x) #all values of x
    o_vals = np.arange(numvals_o)
    z_vals = np.arange(numvals_z)

    pi_t = np.zeros((numvals_x,))
    pi_t[:] = 1/(numvals_x) #start with a uniform belief state (distribution over latent states)
    pi_t_all = np.zeros((numvals_x, T))
    pi_t_all[:,0] = pi_t[:]

    theta_bar_hat_t = np.zeros((theta_bar_dim, T)) #2-dimensional estimate through time
    
    ######################################## INITIAL ESTIMATE OF THETA
    theta_bar_hat_t[:,0]= theta_bar_init # a random initial guess of the unknown parameter
    
    state_dist_shape = (numvals_x, numvals_x, numvals_z) #p(X_{n+1} | X_n, Z_n)
    obs_dist_shape = (numvals_o, numvals_x) #p(O_{n+1} | X_{n+1})
    dist_shape_list = [state_dist_shape, obs_dist_shape]
    theta_indices, theta_not_indices = get_nonzero_indices_state(state_dist_shape, M)
    ind_obs = get_nonzero_indices_obs(obs_dist_shape, K, unsymmetric)
    if not unsymmetric:
        phi_indices, phi_not_indices = ind_obs[0], ind_obs[1]
        theta_bar_indices = [(theta_indices, theta_not_indices), (phi_indices, phi_not_indices)]
    else:
        phi_indices, phi_not_indices, zeta_indices, zeta_not_indices = ind_obs[0][0], ind_obs[1][0], ind_obs[0][1], ind_obs[1][1]
        theta_bar_indices = [(theta_indices, theta_not_indices), (phi_indices, phi_not_indices), (zeta_indices, zeta_not_indices)]
    dist_list = update_dist(theta_bar_hat_t[:,0], dist_shape_list, theta_bar_limits, theta_bar_indices, M, unsymmetric)

    omega_bar_t = np.zeros((theta_bar_dim,)) #initial value
    omega_bar_t_all = np.zeros((theta_bar_dim ,T))
    omega_bar_t_all[:,0]=omega_bar_t
    
    for t in range(1,T):
        #dynamics
        xi_arr[t] = np.random.binomial(1, theta_bar_star[0]) #with prob theta
        X_qlen[t], O_obs[t] = update_X_O(X_qlen[t-1], Z_dep[t-1], xi_arr[t], M, K, theta_bar_star, unsymmetric)
        #control policy
        Z_dep[t] = (1-O_obs[t])*np.random.binomial(1, p=dep_rate[0]) + (O_obs[t])*np.random.binomial(1, p=dep_rate[1])
        ztmin1_ind, ot_ind = get_z_o_indices(Z_dep[t-1], O_obs[t], z_vals, o_vals)
        
        #algorithm
        #omega_update
        tred = math.ceil(t/tau)
        a_t = A/tred**alpha
        b_t = B/tred**beta
        omega_bar_t=update_omega(omega_bar_t, b_t, w0_bar)
        omega_bar_t_all[:,t]=omega_bar_t
        #theta-update
        theta_bar_hat_t[:,t] = update_theta_es(theta_bar_hat_t[:,t-1], dist_shape_list, a_t, omega_bar_t_all[:,t-1], \
                                               pi_t_all[:,t-1], ot_ind, numvals_z, theta_bar_indices, \
                                               dep_rate, theta_bar_limits, M, K, gamma_bar, unsymmetric) 
        dist_list = update_dist(theta_bar_hat_t[:,t], dist_shape_list, theta_bar_limits, theta_bar_indices, M, unsymmetric)
        #pi-update
        # pi_t = update_pi(pi_t, dist_list, ot_ind, numvals_z, dep_rate=dep_rate)
        pi_t_all[:,t] = update_pi_new(pi_t_all[:,t-1], ot_ind, dep_rate, omega_bar_t_all[:,t-1], theta_bar_hat_t[:,t-1], \
                             gamma_bar, dist_shape_list,theta_bar_limits, theta_bar_indices, M, unsymmetric)
        # pi_t_all[:,t] = pi_t[:]
        
    results = {'X':X_qlen, 'O':O_obs, 'xi':xi_arr, 'Z':Z_dep, 'pi_t_all':pi_t_all, 'theta_hat_t':theta_bar_hat_t,\
               'omega_t':omega_bar_t_all, 'dist_list':dist_list}
    return results