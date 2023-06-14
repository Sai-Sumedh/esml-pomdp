import numpy as np
import time, math
import gym

# adapted from _parse_file() in utils.py
def add_new_rule(path_to_file, rule_id, action_attempt, rule_new):
        file = open(path_to_file, 'r+')
        current_section = None
        sections = ['Layout', 'Abstraction', 'Behaviour', 'Rewards']
        replaced_content = ""
        old_rule_exists = False
        data = file.readlines()
        index_line=0
        index_behaviour = []
        for line in data:
            line = line.strip()
            for section_name in sections:
                if section_name in line:
                    current_section = section_name   
            if current_section == 'Behaviour':
                index_behaviour.append(index_line)
                if line.startswith(f'{rule_id}-{action_attempt}'): #the particular rule to be replaced
                    old_rule_exists = True
                    new_line = rule_new
                else:
                    new_line = line
                replaced_content = replaced_content + new_line + '\n'
            else:
                new_line = line
                replaced_content = replaced_content + new_line + '\n'
            index_line += 1
        file.close()
        if old_rule_exists: #need to replace the old rule
            with open(path_to_file, 'w') as f:
                f.write(replaced_content)
        else: #need to append the new rule to the file
            # with open(path_to_file, 'a+') as f:
            #     f.write('\n')
            #     f.write(rule_new+'\n')
            # print(index_behaviour)
            # print(len(data))
            data.insert(min(index_behaviour[-1], len(data)-1), rule_new+'\n')
            with open(path_to_file, 'w') as f:
                contents = "".join(data)
                f.write(contents)

def add_rule_to_world(theta_star, world_to_mimic, new_world_name='world_new', rule_id_new='1', action_attempt_new = 3 , \
                     act_prob_pairs_new = None):
    #create a new world by making a copy of world_new
    import shutil
    shutil.copyfile(f'./gridworld-gym-master/worlds/{world_to_mimic}.txt', f'./gridworld-gym-master/worlds/{new_world_name}.txt')
    
    if act_prob_pairs_new is None:
        act_prob_pairs_new = [(3, theta_star), (1, 1-theta_star)]
    actions_dict = {0:'up', 1:'down', 2:'left', 3:'right', -1: 'none'}
    if isinstance(action_attempt_new, int): #only one new rule
        rule_new_text = f"{rule_id_new}-{actions_dict[action_attempt_new]}-[{actions_dict[act_prob_pairs_new[0][0]]}:{act_prob_pairs_new[0][1]}, {actions_dict[act_prob_pairs_new[1][0]]}:{act_prob_pairs_new[1][1]}]"
        add_new_rule(f'./gridworld-gym-master/worlds/{new_world_name}.txt', rule_id_new, actions_dict[action_attempt_new], rule_new_text)

    elif isinstance(action_attempt_new, list): #list of new rules
        if not isinstance(rule_id_new, list):
            rule_id_new2 = [rule_id_new for _ in range(len(action_attempt_new))]
        for i in range(len(action_attempt_new)):
            rule_new_text = f"{rule_id_new2[i]}-{actions_dict[action_attempt_new[i]]}-[{actions_dict[act_prob_pairs_new[i][0][0]]}:{act_prob_pairs_new[i][0][1]}, {actions_dict[act_prob_pairs_new[i][1][0]]}:{act_prob_pairs_new[i][1][1]}]"
            add_new_rule(f'./gridworld-gym-master/worlds/{new_world_name}.txt', rule_id_new2[i], actions_dict[action_attempt_new[i]], rule_new_text)

    rule_new_details = {'rule_id':rule_id_new, 'action':action_attempt_new, 'act_prob_pairs':act_prob_pairs_new}

    return rule_new_details
                
def update_loc(old_loc, action, latent_space):
    """
    old_loc: tuple, 
    action: in {0,1,2,3,-1}
    """
    if action==0:
        new_loc = old_loc[0]-1, old_loc[1]
    elif action==1:
        new_loc = old_loc[0]+1, old_loc[1]
    elif action==2:
        new_loc = old_loc[0], old_loc[1]-1
    elif action==3:
        new_loc = old_loc[0], old_loc[1]+1
    elif action==-1:
        new_loc = old_loc[0], old_loc[1]
    
    if latent_space[new_loc[0],new_loc[1]] == '#':
        new_loc = old_loc[0],old_loc[1] # wall, so new location same as the previous one
    return new_loc

def initialize_p_theta(latent_space, observation_space, action_space, env, abstract_world, rule_new_details, theta_init=0):
    
    stochastic_tiles_list = list(env.stochastic_tile.keys())
    
    p_theta = np.zeros((*latent_space.shape, *observation_space.shape, *latent_space.shape, *action_space.shape))
    #fill up p_theta for deterministic transitions (this needs to be done only once)
    for i in range(latent_space.shape[0]):
        for j in range(latent_space.shape[1]):
            for k in range(action_space.shape[0]):
                #we will never start from a 'wall' location, so ignore those entries
                if latent_space[i,j]!='#':
                    #first update for deterministic tiles
                    # (i,j) is either not a stochastic tile, or it is but the action is not part of the rule (hence not stochastic)
                    if (i,j) not in stochastic_tiles_list or action_space[k] not in env.rules[env.stochastic_tile[(i,j)]].behaviour.keys():
                        new_loc = update_loc((i,j), action_space[k], latent_space)
                        new_obs = abstract_world[new_loc[0], new_loc[1]]
                        new_obs_index = int(np.argwhere(observation_space==new_obs)) 
                        p_theta[new_loc[0], new_loc[1], new_obs_index, i, j, k] = 1 #deterministic transition
                   
    
    # indices of scalar parameter theta, to make it easier to compute p^(\phi) at each timestep
    theta_indices = np.zeros(p_theta.shape, dtype=bool) # indices of theta, to ensure easy computation of the transition probability
    theta_not_indices = np.zeros(p_theta.shape, dtype=bool) #indices of 1-theta

    # for stochastic tiles, it suffices to go over particular tiles and corresponding actions
    for loc in env.stochastic_tile.keys():
        rule_id = env.stochastic_tile[loc]
        actions_stoch_behav = list(env.rules[rule_id].behaviour.keys())
        p_theta[:,:,:,loc[0], loc[1],actions_stoch_behav] = 0 #start with 0s for all stochastic tiles
        #actions which lead to stochastic results
        for k in range(len(actions_stoch_behav)): 
            #resulting actions and corresponding probabilities
            for act_prob_pair in env.rules[rule_id].behaviour[action_space[k]]:
                new_loc = update_loc(loc, act_prob_pair[0], latent_space)
                new_obs = abstract_world[new_loc[0], new_loc[1]]
                new_obs_index = int(np.argwhere(observation_space==new_obs)) 
                
                # need to add when multiple probabilistic actions lead to wall, resulting in the same state again
                if isinstance(rule_new_details['action'],int):
                    if rule_id==rule_new_details['rule_id'] and action_space[k]==rule_new_details['action']:# and act_prob_pair in rule_new_details['act_prob_pairs']:
                        # print("condition met")
                        if act_prob_pair[0]==rule_new_details['act_prob_pairs'][0][0]: #same future state, to get theta's location
                            # print("condition2 met")
                            theta_indices[new_loc[0], new_loc[1], new_obs_index, loc[0], loc[1], k] = 1
                        elif act_prob_pair[0]==rule_new_details['act_prob_pairs'][1][0]:
                            # print("condition3 met")
                            theta_not_indices[new_loc[0], new_loc[1], new_obs_index, loc[0], loc[1], k] = 1
                        else:
                            raise Exception("probabilities in the world don't match entered values")
                    else: #stochastic tile with fixed probability (independent of theta_star)
                        p_theta[new_loc[0], new_loc[1], new_obs_index, loc[0], loc[1], k] = act_prob_pair[1]
                        # print("hi")
                elif isinstance(rule_new_details['action'], list):
                    for m in range(len(rule_new_details['action'])):
                        if rule_id==rule_new_details['rule_id'] and action_space[k]==rule_new_details['action'][m]:# and act_prob_pair in rule_new_details['act_prob_pairs']:
                            # print("condition met")
                            if act_prob_pair[0]==rule_new_details['act_prob_pairs'][m][0][0]: #same future state, to get theta's location
                                # print("condition2 met")
                                theta_indices[new_loc[0], new_loc[1], new_obs_index, loc[0], loc[1], k] = 1
                            elif act_prob_pair[0]==rule_new_details['act_prob_pairs'][m][1][0]:
                                # print("condition3 met")
                                theta_not_indices[new_loc[0], new_loc[1], new_obs_index, loc[0], loc[1], k] = 1
                            else:
                                raise Exception("probabilities in the world don't match entered values")
                        else: #stochastic tile with fixed probability (independent of theta_star)
                            p_theta[new_loc[0], new_loc[1], new_obs_index, loc[0], loc[1], k] = act_prob_pair[1]
                            # print("hi")
               
    eps=1e-5
    if theta_init>1-eps:
        theta_init = 1-eps
    elif theta_init<eps:
        theta_init = eps
    p_theta[theta_indices] =theta_init
    p_theta[theta_not_indices] = 1-theta_init
    return p_theta, theta_indices, theta_not_indices

def update_pi_fixedparam(pi_curr, p_theta, Onew_ind, numvals_z):
    
    prod_pi_ptheta = np.tensordot(pi_curr, np.sum(p_theta[:,:,Onew_ind,:,:,:]*(1/numvals_z), axis=-1), axes=([0,1],[2,3])) #sum of products over Xnmin1, after marginalizing over Z
    if np.sum(prod_pi_ptheta.flatten())==0:
        print(prod_pi_ptheta)
        print(f"O={Onew_ind}, Z={Zprev_ind}")
    pi_new = prod_pi_ptheta/np.sum(prod_pi_ptheta.flatten())
    assert np.array_equal(pi_curr.shape, pi_new.shape)
    return pi_new

def update_pi_new(pi_curr, p_theta, Onew_ind, numvals_z, theta_t, gamma, omega_t, theta_indices, theta_not_indices):
    
    p_theta_pert = compute_p_theta(p_theta, theta_t+gamma*np.sin(omega_t), theta_indices, theta_not_indices)
    prod_pi_ptheta = np.tensordot(pi_curr, np.sum(p_theta_pert[:,:,Onew_ind,:,:,:]*(1/numvals_z), axis=-1), axes=([0,1],[2,3])) 
    
    if np.sum(prod_pi_ptheta.flatten())==0:
        print(prod_pi_ptheta)
        print(f"O={Onew_ind}, Z={Zprev_ind}")
    pi_new = prod_pi_ptheta/np.sum(prod_pi_ptheta.flatten())
    assert np.array_equal(pi_curr.shape, pi_new.shape)
    return pi_new

def get_z_o_indices(zk, ok, z_vals, o_vals): #dont need this, since outputs of env are already indices
    zind = int(np.argwhere(z_vals==zk))
    oind = int(np.argwhere(o_vals==ok))
    return zind, oind

def compute_p_theta(p_theta_old, theta_new, theta_indices, theta_not_indices):
    """
    Idea: pre-identify indices where p appears and where 1-p appears
    to update the transition kernel, just update the saved indices 
    directly instead of looping over XZ values again (most are zeroes)
    """
    # p_theta = np.zeros(p_theta_old.shape)
    p_theta = np.copy(p_theta_old) #to retain the 'fixed' probabilities
    eps=1e-5
    if theta_new<eps:
        theta_new = eps
    elif theta_new>1-eps:
        theta_new=1-eps
    # theta_old = np.amin(p_theta_old[theta_indices]) #min to account for any additions to theta at some locations due to wall
    # p_theta[theta_indices] = p_theta_old[theta_indices]-theta_old+theta_new
    # p_theta[theta_not_indices] = p_theta_old[theta_not_indices]+theta_old-theta_new
    p_theta[theta_indices] = theta_new
    p_theta[theta_not_indices] = 1-theta_new
    return p_theta

def update_omega(omega, b_t, eps=0.1):
    return omega*(1)+eps*b_t #linear increase instead of exponential

def update_theta_es(theta_old, p_theta_old, a_t, omega_t, pi_t, Onew_ind, numvals_z, theta_indices, theta_not_indices, gamma=0.1):
    #compute p_theta at a perturbed value of theta
    p_theta_pert = compute_p_theta(p_theta_old, theta_old+gamma*np.sin(omega_t), theta_indices, theta_not_indices)
    p_theta_pert_marg = np.sum(p_theta_pert[:,:,Onew_ind,:,:,:]*(1/numvals_z), axis=(0,1,4)) #marginalize over X_n+1 and Z_n
    p_th_pert_cond = np.tensordot(pi_t, p_theta_pert_marg, axes=([0,1],[0,1]))
    thr = 1e-5
    theta_new = theta_old + a_t*(1/gamma)*np.sin(omega_t)*np.log(p_th_pert_cond)
        
    thr=1e-5
    if theta_new<thr:
        theta_new = thr
    elif theta_new>1-thr:
        theta_new=1-thr
    return theta_new


def estimate_theta_es_gridworld(theta_init, theta_star, T=500000, alpha = 1, beta = 0.8, eps = 0.1, \
                                A=0.1, B=10, tau=100, gamma=0.1,\
                                world_to_mimic='world0', new_world_name='world_new',\
                               rule_id_new=None, action_attempt_new = None , act_prob_pairs_new = None):
    #add new rule with theta_star
    if rule_id_new is not None and action_attempt_new is not None and act_prob_pairs_new is not None:
        rule_new_details = add_rule_to_world(theta_star, world_to_mimic=world_to_mimic, new_world_name=new_world_name,\
                                            rule_id_new=rule_id_new, action_attempt_new=action_attempt_new, act_prob_pairs_new=act_prob_pairs_new)
    else:
        rule_new_details = add_rule_to_world(theta_star, world_to_mimic=world_to_mimic, new_world_name=new_world_name)
    #create environment
    env = gym.make(id='poge-v1', 
                   world_file_path=f'gridworld-gym-master/worlds/{new_world_name}.txt',
                   force_determinism=False,
                   indicate_slip=False,
                   is_partially_obs=True,
                   one_time_rewards=True,
                   step_penalty=-0.1)
    #get spaces for X, O, Z
    latent_space = np.array(env.world)
    abstract_world = np.array(env.abstract_world)
    observation_space = np.array(list(env.state_2_one_hot_map.keys())) #the observation space match the env.state_2_one_hot_map() which decides the observation output after each step
    action_space = np.array(env.actions)
    stochastic_tiles_list = list(env.stochastic_tile.keys())
    #define belief state
    pi_t = np.zeros(latent_space.shape)
    pi_t[:] = 1/np.size(pi_t) #distribution is initially uniform
    pi_t_all = np.zeros((*latent_space.shape, T))
    pi_t_all[:,:,0] = pi_t
    #evolution of parameter estimate in time
    theta_hat_t = np.zeros((T,)) #estimate through time
    theta_hat_t[0]= theta_init # theta_init is given as input to this function

    p_theta_hat, theta_indices, theta_not_indices = initialize_p_theta(latent_space, observation_space, action_space, env, \
                                                                   abstract_world, rule_new_details, theta_init)
    
    # omega_t = 0.1 #initial value
    omega_t = 0 #initial value
    omega_t_all = np.zeros((T,))
    omega_t_all[0]=omega_t

    env.action_space.seed(42)
    env.reset()
    
    Ot_all = np.zeros((T,))
    Xt_all = [None]*T
    Xt_all[0] = env.player_location
    Ot_all[0] = int(np.argwhere(observation_space == abstract_world[Xt_all[0]]))
    Zt_all = np.zeros((T,), dtype=np.int8)

    tic = time.perf_counter()
    for t in range(1, T):
        Zt = np.random.choice(action_space) #policy: choose Z uniformly at random
        Zt_all[t]=Zt
        Ot, reward, terminated, info = env.step(Zt)
        Xt_all[t] = env.player_location
        Ot_all[t] = Ot
        tred = math.ceil(t/tau) #tau- number of steps for which a_t, b_t remain fixed
        a_t = A/tred**alpha #A is a tunable parameter
        b_t = B/tred**beta #B is also a tunable parameter
        omega_t=update_omega(omega_t, b_t, eps)
        omega_t_all[t]=omega_t
        theta_hat_t[t] = update_theta_es(theta_old=theta_hat_t[t-1], p_theta_old=p_theta_hat, a_t=a_t, \
                                         omega_t=omega_t_all[t-1], pi_t=pi_t_all[:,:,t-1], Onew_ind=Ot, \
                                         numvals_z=len(action_space), theta_indices=theta_indices, theta_not_indices=theta_not_indices, \
                                        gamma=gamma)  
        p_theta_hat = compute_p_theta(p_theta_hat, theta_hat_t[t], theta_indices, theta_not_indices)
        # pi_t = update_pi(pi_t, p_theta_hat, Ot, numvals_z=len(action_space))
        # pi_t_all[:,:,t] = pi_t 
        pi_t_all[:,:,t] = update_pi_new(pi_t_all[:,:,t-1], p_theta_hat, Ot, len(action_space), \
                                        theta_hat_t[t-1], gamma, omega_t_all[t-1], theta_indices, theta_not_indices)
    toc = time.perf_counter()
    runtime = toc-tic
    # print(f"For {T} timesteps, time taken={runtime}s  ({runtime/T}s per timestep)")
    env.close()
    
    results = {'X':Xt_all, 'O':Ot_all, 'Z':Zt_all, 'pi_t_all':pi_t_all, 'theta_hat_t':theta_hat_t,\
               'omega_t':omega_t_all, 'p_theta':p_theta_hat, 'theta_indices':theta_indices, \
               'theta_not_indices':theta_not_indices, 'env':env, 'rule_new_details':rule_new_details}
    return results


def compute_ll_next(ll_prev, pi_t, p_theta, Onew_ind, numvals_z):
    p_theta_marg = np.sum(p_theta[:,:,int(Onew_ind),:,:,:]*(1/numvals_z), axis=(0,1,4)) #marginalize over X_n+1 and Z_n
    p_th_cond = np.tensordot(pi_t, p_theta_marg, axes=([0,1],[0,1]))
    ll_next = ll_prev + np.log(p_th_cond)
    return ll_next


def compute_log_likelihood_alldata(theta_estimate, O_t, Z_t, Tmax, env, rule_new_details, saveLLt=False):
    LL = 0
    
    latent_space = np.array(env.world)
    abstract_world = np.array(env.abstract_world)
    observation_space = np.array(list(env.state_2_one_hot_map.keys())) #the observation space match the env.state_2_one_hot_map() which decides the observation output after each step
    action_space = np.array(env.actions)
    
    numvals_z = len(action_space)
    p_theta, theta_indices, theta_not_indices = initialize_p_theta(latent_space, observation_space, action_space, env, abstract_world, rule_new_details, theta_estimate)
    
    pi_t = np.zeros(latent_space.shape)
    pi_t[:] = 1/(np.size(latent_space)) #start with a uniform belief state (distribution over latent states)
    LL_t = np.zeros((Tmax,))
    for t in range(1,Tmax):
        LL = compute_ll_next(ll_prev=LL, pi_t=pi_t, p_theta=p_theta, Onew_ind=O_t[t], numvals_z=numvals_z) #when O_t is already an index
        pi_t = update_pi_fixedparam(pi_t, p_theta, int(O_t[t]), numvals_z=len(action_space))
        LL_t[t] = LL
    if saveLLt:
        return LL_t
    else:
        return LL