import stable_baselines3
import numpy as np
import pandas as pd
import matplotlib
import os
import json
import torch
import time
from RL_policy.RL_environments import CSTR_sp
from backup_policy.auxiliary_functions_NN import ARMPC_CSTR

matplotlib.use('TkAgg')

def closed_loop_evaluation(dir: str = None, num_eval: int = 50, save_results: bool = False, single_loop: bool = False, seed: int = 123, sp: bool= True, RL_agent:bool = True, feas_IC:bool = False, sampling: str = 'uniform'):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if RL_agent:
        with open(dir + '\\setting_dict.json') as f:
                hyperparameters = json.load(f)
 
        model = stable_baselines3.TD3.load(rf'{dir}\best_model\best_model.zip',
        custom_objects={"buffer_size": 100000})
        model = model.policy
        model = model.to(device)

    else: 
        # Load backup policy for evaluation
        with open(os.path.join(dir, 'hyperparameters.json'), 'r') as f:
            hyperparameters = json.load(f)
        model = ARMPC_CSTR(input_size = hyperparameters['input_size'], hidden_size = hyperparameters['num_neurons'], num_hidden_layers = hyperparameters['num_hidden_layers'], layer_act = hyperparameters['activation_function'])
        model.load_state_dict(torch.load(os.path.join(dir, 'best_model.pth'), weights_only=True, map_location=device))
        model.to(device)
        model.eval()
    
    env = CSTR_sp(seed = seed, fix_p_unc = False, feas_IC = feas_IC)

    CV_s_list= []
    reward_mpc_list = []        # MPC objective without consideration of CV
    pred_time_list = []
    CV_all_list = []
    p1 = []
    p2 = []
    safety_parameter = []
    
    np.random.seed(seed)
    for i in range(num_eval):
        obs,_ = env.reset(seed = i, sampling = sampling)

        CV_all = 0
        CV_s = 0
        mpc_reward = 0
        states_list = []
        action_list = []
        terminated = False
        pred_time = 0
        number_steps = 0

        states = env.state_phy

        while not terminated:

            states_list.append(states)
            if not sp: 
                obs = obs[:-1]
            
            start_time = time.perf_counter()
            action = model(torch.tensor(obs.reshape(1,-1), dtype = torch.float32).to(device)).cpu().detach().numpy()
            action = np.clip(action, -1, 1)     # ensure input constraint satisfaction
            pred_time += time.perf_counter() - start_time

            obs, _, terminated,_, info = env.step(action)
            states = env.state_phy
            action_list.append(env.action_num)

            if env._check_constraint_violation(states):
                CV_all += 1
                CV_s += env.constraint_deviation(states)
            mpc_reward += info['reward_MPC']
            number_steps += 1

        reward_mpc_list.append(mpc_reward)
        pred_time_list.append((pred_time/number_steps)*1000)    # average time per step in ms
        CV_all_list.append(CV_all)
        CV_s_list.append(CV_s)
        p1.append(env.p[0])
        p2.append(env.p[1])
        safety_parameter.append(env.safety_parameter)
    
    if single_loop:
        traj_s = pd.DataFrame(np.squeeze(states_list), columns=['C_a', 'C_b', 'T_R', 'T_K'])
        traj_a = pd.DataFrame(np.squeeze(action_list), columns=['F', 'Q_dot'])
        return traj_s, traj_a
    
    # Save the results in dataframe
    results = pd.DataFrame({'mpc_reward':reward_mpc_list
                            , 'CV_all': CV_all_list
                            ,'CV_s':CV_s_list
                            , 'pred_time':pred_time_list
                            , 'safety_parameter': safety_parameter
                            , 'alpha': p1,'beta': p2})
    
    if save_results:
        results.to_excel(fr'{dir}\\only_RL_closed_loop_eval.xlsx')
    
    return results
    

def eval_ensemble_agents(dir: str, num_models: int, num_eval: int = 50, sp: bool = True, RL_agent: bool = True, feas_IC: bool = False, sampling = 'uniform'):
    # Initialize a DataFrame to store mean results for all models
    all_mean_results = pd.DataFrame()

    for i in range(num_models):
        if RL_agent:
            dir_agent = dir + f'\model_{i}'
        else:
            dir_agent = dir + f'\grid{i+1}'
        
        # check whether agent_dir exists
        if not os.path.exists(dir_agent):
            print(f"Directory {dir_agent} does not exist. Skipping.")
            continue

        results = closed_loop_evaluation(dir_agent, num_eval=num_eval, save_results=True, sp = sp, RL_agent = RL_agent, feas_IC= feas_IC, sampling = sampling)

        # Compute the mean of the current results
        mean_results = results.mean().to_frame().T  # Convert to DataFrame with a single row
        
        if RL_agent:
            mean_results.insert(0, 'model', f'model_{i}') 
        else:
            mean_results.insert(0, 'grid', f'grid_{i+1}')

        # Concatenate with the existing DataFrame
        all_mean_results = pd.concat([all_mean_results, mean_results], ignore_index=True)
           

    # Save the concatenated mean results to an Excel file
    all_mean_results.to_excel(fr'{dir}\only_RL_closed_loop_eval_all_models.xlsx', index=False)

    return all_mean_results

def monotonicity_check(agent_dir: str, num_models: int, episode_length: int = 100):
    cv_all_results = []
    cv_s_results = []
    for l in range(num_models):
        model = stable_baselines3.TD3.load(rf'{agent_dir}\model_{l}\best_model\best_model.zip',
        custom_objects={"buffer_size": 100000})
        cv_all_row = {'Model': f'model_{l}'}
        cv_s_row = {'Model': f'model_{l}'}
        print(f'Evaluating model {l}')

        for sigma in np.arange(0, 0.21, 0.01):
            CV_all, CV_s= closed_loop_evaluation_fixed_sigma(sigma=sigma, model = model, episode_length=episode_length)
            cv_all_row[f'{sigma:.2f}'] = CV_all
            cv_s_row[f'{sigma:.2f}'] = CV_s
    
        cv_all_results.append(cv_all_row)
        cv_s_results.append(cv_s_row)


    df_cv_all = pd.DataFrame(cv_all_results)
    df_cv_s = pd.DataFrame(cv_s_results)
    
    with pd.ExcelWriter(os.path.join(agent_dir, f'monotonicity_check_results_{episode_length}.xlsx')) as writer:
        df_cv_all.to_excel(writer, sheet_name='CV_all', index=False)
        df_cv_s.to_excel(writer, sheet_name='CV_s', index=False)


def closed_loop_evaluation_fixed_sigma(sigma: float, model, episode_length:int = 100, num_eval: int = 100, seed: int = 123):
    env = CSTR_sp(seed = seed, fix_p_unc = False)
    CV_s_list= []
    CV_all_list = []
    
    np.random.seed(123)
    for i in range(num_eval):
        obs,_ = env.reset(seed = i
                          )
        # fix sigma 
        obs_sp = obs
        obs_sp[-1] = sigma
        obs[-1] = env.observation_scaler.transform(obs_sp.reshape(1,-1)).reshape(-1,1)[-1] 

        CV_all = 0
        CV_s = 0

        states = env.state_phy
        for _ in range(episode_length):
            action, _states = model.predict(obs)

            obs, reward, terminated,_, info = env.step(action)
            
            # fix sigma 
            obs_sp = obs
            obs_sp[-1] = sigma
            obs[-1] = env.observation_scaler.transform(obs_sp.reshape(1,-1)).reshape(-1,1)[-1]

            states = env.state_phy

            if env._check_constraint_violation(states):
                CV_all += 1
                CV_s += env.constraint_deviation(states)
        
        CV_all_list.append(CV_all)
        CV_s_list.append(CV_s)

    return np.mean(CV_all_list), np.mean(CV_s_list)