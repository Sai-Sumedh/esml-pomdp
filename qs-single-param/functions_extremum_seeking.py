import numpy as np
from functions_queueing_system import *

def update_omega(omega, b_t, eps=0.1):
    return omega*(1)+eps*b_t #linear increase instead of exponential


def update_theta_es(theta_old, p_theta_old, a_t, omega_t, pi_t, ot_ind, numvals_z, theta_indices, theta_not_indices, M, K, gamma=0.1, 
                   del_theta_prev=None, dep_rate=None, poisson_arrivals_departures=False, dep_dists=None):
    """
    Includes a departure rate (prob of departure), and momentum
    """
    p_th_pert = compute_p_theta(p_theta_old, theta_old+gamma*np.sin(omega_t), theta_indices, theta_not_indices, M, K)
    # if dep_rate is None:
    #     p_th_pert_xn = np.sum(p_th_pert[:, ot_ind, :, :], axis=(0,2))/numvals_z #marginalize over X_{n+1} and Z_{n}, |A|=2
    # else:
    #     p_th_pert_xn = np.sum(np.sum(p_th_pert[:, ot_ind, :, :], axis=0)*np.array([1-dep_rate, dep_rate]).reshape((1,-1)),axis=1) #marginalize over X_{n+1} and Z_{n}, |A|=2
    #ot_ind=0-> O=0, for which dep_rate[0] is correct; ot_ind=1->O=1 for which dep_rate[1] is correct
    if poisson_arrivals_departures:
        p_th_pert_xn = np.sum(np.sum(p_th_pert[:, ot_ind, :, :], axis=0)*dep_dists[ot_ind].reshape((1,-1)),axis=1)
    else:
        p_th_pert_xn = np.sum(np.sum(p_th_pert[:, ot_ind, :, :], axis=0)*np.array([1-dep_rate[ot_ind],\
                                                                               dep_rate[ot_ind]]).reshape((1,-1)),axis=1)
    p_th_pert_cond = np.sum(pi_t.reshape((-1,))*p_th_pert_xn.reshape((-1,))) #p(O_{t}\mid O^{t-1})
    if del_theta_prev is None:
        theta_new = theta_old + a_t*np.sin(omega_t)*np.log(p_th_pert_cond)
    else:
        eta = 0.5
        theta_new = theta_old + a_t*np.sin(omega_t)*np.log(p_th_pert_cond) + eta*del_theta_prev
    thr = 1e-5 #threshold- to limit theta to (thr, 1-thr)
    if theta_new<thr:
        theta_new = thr
    elif theta_new>1-thr:
        theta_new=1-thr
    return theta_new


def estimate_theta_es(theta_init, theta_star, M = 20, K = 3, T = 2000,\
                                    alpha=0.8, c=1, beta=0.7, eps=1, gamma=0.1,\
                      momentum=False, dep_rate=[0.1, 0.7], poisson_arrivals_departures=False, dep_dists=None,\
                     gamma_decay=False):
    
    numvals_x = M+1
    numvals_o = 2 #binary

    X_qlen = np.zeros((T,))
    O_obs = np.zeros((T,))
    xi_arr = np.zeros((T,))
    Z_dep = np.zeros((T,))
    
    if poisson_arrivals_departures:
        numvals_z = M+1
        dep_dist0 = np.array([(dep_rate[0]**k)*np.exp(-dep_rate[0])/(np.math.factorial(k)) for k in range(numvals_z)])
        dep_dist0 = dep_dist0/np.sum(dep_dist0)

        dep_dist1 = np.array([(dep_rate[1]**k)*np.exp(-dep_rate[1])/(np.math.factorial(k)) for k in range(numvals_z)])
        dep_dist1 = dep_dist1/np.sum(dep_dist1)

        dep_dists = [dep_dist0, dep_dist1]

        arr_dist = np.array([(theta_star**k)*np.exp(-theta_star)/(np.math.factorial(k)) for k in range(numvals_x)])
        arr_dist = arr_dist/np.sum(arr_dist)
    else: 
        numvals_z = 2
    
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
    gamma_all = []
    for t in range(1,T):
        #dynamics of the process
        if poisson_arrivals_departures:
            xi_arr[t] = np.random.choice(np.arange(numvals_x), p=arr_dist)
        else:
            xi_arr[t] = np.random.binomial(1, theta_star)
        
        X_qlen[t], O_obs[t] = update_X_O(X_qlen[t-1], Z_dep[t-1], xi_arr[t], M, K)

        #control choice - random for now
        # if dep_rate is None:
        #     Z_dep[t] = np.random.randint(0, numvals_z)
        # else: #assumes Z=0 or 1, distributed as bernoulli(dep_rate)
        #     Z_dep[t] = np.random.binomial(1, dep_rate)
        if poisson_arrivals_departures:
            if O_obs[t]==0:
                Z_dep[t] = np.random.choice(np.arange(numvals_z), p=dep_dist0)
            elif O_obs[t]==1:
                Z_dep[t] = np.random.choice(np.arange(numvals_z), p=dep_dist1)
        else:
            Z_dep[t] = (1-O_obs[t])*np.random.binomial(1, p=dep_rate[0]) + (O_obs[t])*np.random.binomial(1, p=dep_rate[1])

        ztmin1_ind, ot_ind = get_z_o_indices(Z_dep[t-1], O_obs[t], z_vals, o_vals)
        tred = math.ceil(t/100)
        
        # a_t = 0.1/tred**alpha #works beautifully
        # a_t = 0.1/tred**alpha
        a_t = 0.1/tred**alpha
        b_t = 10/tred**beta
        
        omega_t=update_omega(omega_t, b_t, eps)
        omega_t_all[t]=omega_t
        if t>=2 and momentum:
            del_theta_prev=theta_hat_t[t-1]-theta_hat_t[t-2]
        else:
            del_theta_prev = None
        if gamma_decay:
            exponent_gamma = 1
            gamma_new = gamma/tred**exponent_gamma
        else:
            gamma_new = gamma
        gamma_all.append(gamma_new)
        theta_hat_t[t] = update_theta_es(theta_hat_t[t-1], p_theta, a_t, omega_t, pi_t, ot_ind, numvals_z, theta_indices, \
                                         theta_not_indices, M, K, gamma_new, del_theta_prev = del_theta_prev, dep_rate=dep_rate,\
                                        poisson_arrivals_departures=poisson_arrivals_departures, dep_dists=dep_dists) 
        p_theta = compute_p_theta(p_theta, theta_hat_t[t], theta_indices, theta_not_indices, M, K)
        pi_t = update_pi(pi_t, p_theta, ot_ind, numvals_z, dep_rate=dep_rate, poisson_arrivals_departures=poisson_arrivals_departures,\
                        dep_dists=dep_dists)
        pi_t_all[:,t] = pi_t[:]
        
    results = {'X':X_qlen, 'O':O_obs, 'xi':xi_arr, 'Z':Z_dep, 'pi_t_all':pi_t_all, 'theta_hat_t':theta_hat_t,\
               'omega_t':omega_t_all, 'p_theta':p_theta, 'gamma_t':gamma_all}
    return results