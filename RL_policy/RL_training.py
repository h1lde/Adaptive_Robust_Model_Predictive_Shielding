import os
import json
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList

from RL_policy.RL_environments import CSTR_sp, CSTR_base
from RL_policy.auxiliary_functions_RL import TD3SchedulerCallback

def RL_training(setting_dict: dict = None):
    
    # read parameters from dict
    seed = setting_dict['seed']
    save_dir = setting_dict['save_dir'] + fr'/model_{seed-48}'
    
    os.makedirs(save_dir, exist_ok=True)

    # Create environment with or without safety parameter
    if setting_dict['env_type'] == 'CSTR_sp':
        env = DummyVecEnv([lambda: Monitor(CSTR_sp(seed = seed, penalty_weight = setting_dict['penalty_weight'], max_sp = setting_dict['max_sp']), save_dir)])
        env_eval = CSTR_sp(seed = seed  + 100, penalty_weight = setting_dict['penalty_weight'], max_sp = setting_dict['max_sp'])
    
    elif setting_dict['env_type'] == 'CSTR_base':
        env = DummyVecEnv([lambda: Monitor(CSTR_base(seed = seed, penalty_weight = setting_dict['penalty_weight'], max_sp = setting_dict['max_sp']), save_dir)])
        env_eval = CSTR_base(seed = seed  + 100, penalty_weight = setting_dict['penalty_weight'], max_sp = setting_dict['max_sp'])
    
    env.reset()
    
    # instantiate the evaluation environment
    save_dir_eval = save_dir + fr'/eval_mon'
    os.makedirs(save_dir_eval, exist_ok=True)
    env_eval.reset()
    env_eval = Monitor(env_eval, save_dir_eval)


    # action noise
    n_actions = env.action_space.shape[-1]
    if setting_dict['action_noise_type'] == 'Normal':
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=setting_dict['action_noise_sigma']*np.ones(n_actions))
    
    elif setting_dict['action_noise_type'] == 'OU':
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=setting_dict['action_noise_sigma']*np.ones(n_actions))


    # adjusted actor and critic networks
    policy_kwargs = dict(
    net_arch=setting_dict['net']  
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # instantiate the algorithm

    model = TD3("MlpPolicy",
        env,
        gamma = setting_dict['gamma'],
        verbose = setting_dict['verbose'],
        batch_size = setting_dict['batch_size'],
        learning_rate = setting_dict['learning_rate'],
        action_noise = action_noise,
        policy_kwargs=policy_kwargs,
        device = device,
        seed = seed, 
        tau=setting_dict['tau'],
        buffer_size=setting_dict['size_of_replay_buffer'],
        policy_delay=setting_dict['policy_delay']
    )

    scheduler_callback = TD3SchedulerCallback(
        lr_start = setting_dict['learning_rate'],  # Use the same value you set in the model
        lr_end = setting_dict['learning_rate'] / 10,  # Target learning rate (e.g., 1e-5 if starting with 1e-4)
        noise_start = setting_dict['action_noise_sigma'],  # Use the same value you set in action_noise
        noise_end = setting_dict['action_noise_sigma'] / 10,  # Target noise level (e.g., 0.01 if starting with 0.1)
        max_timesteps = setting_dict['timesteps']*0.8,  # Your total training duration
        warmup_steps=setting_dict['timesteps'] // 10,  # Warmup for 10% of total steps
        annealing_type = setting_dict['scheduling'],
        verbose = 0
    )


    # Initialize the evaluation callback
    eval_callback = EvalCallback(
        env_eval, 
        best_model_save_path = save_dir + r'/best_model', 
        log_path = save_dir, 
        eval_freq = setting_dict['eval_freq'], 
        deterministic = True, 
        render = False)
    
    callback_list = CallbackList([eval_callback
       , scheduler_callback
            ])

    # Train the agent
    model.learn(total_timesteps = setting_dict['timesteps'], 
                callback = callback_list,
                log_interval = 100)
    

    # Save setting dict
    with open(save_dir + r'/setting_dict.json', 'w') as f:
        json.dump(setting_dict, f)  


    # Save the final model
    model.save(save_dir + r'/final_model')
    model.save_replay_buffer(save_dir + r'/replay_buffer')
    env.close()
