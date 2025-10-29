import stable_baselines3
import torch
import numpy as np
import json
import os
import gymnasium as gym
import pandas as pd
import time

from backup_policy.auxiliary_functions_NN import ARMPC_CSTR, TorchMinMaxScaler
from policy_deployment.torch_model_CSTR import CSTR_model_torch, CSTR_simulator_torch
from msMPC.do_mpc_msMPC import CSTR_model, CSTR_msMPC, CSTR_simulator
from RL_policy.RL_environments import integration_setting

np.random.seed(123)


class shielded_deployment():
    def __init__(self, agent_dir: str = None, env: gym.Env = None, backup_dir:str = None, seed: int = 123, max_sp:float = 0.2, sp: bool = False, adaptive: bool = True, n_horizon: int = 20, **kwargs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agent_dir = agent_dir
        self.env = env
        self.seed = seed
        self.max_sp = max_sp
        self.sp = sp
        self.adaptive = adaptive       # for evaluation with non-adaptive robust shield of agents with safety parammeter 
   

        # precompute safety bounds according to the environment 
        self.lbx_tensor = torch.tensor(self.env.lbx_cv[:4], device=self.device, dtype=torch.float32).reshape(1, 4)
        self.ubx_tensor = torch.tensor(self.env.ubx_cv[:4], device=self.device, dtype=torch.float32).reshape(1, 4)

        if backup_dir is not None:
            self.backup_dir = backup_dir
            self._setup_backup_policy()

        self.n_horizon = n_horizon
        self.integration_setting = integration_setting()
        
        # Set up only our components, not the backup policy
        self._setup_agent()
        self._setup_simulator()
        self._initialize_scaler_torch()


    def _setup_simulator(self):
        # simulator for robust rollouts
        self.CSTR_model = CSTR_model_torch()
        self.CSTR_simulator = CSTR_simulator_torch(model = self.CSTR_model, dt = self.env.integration_setting.dt)
        self.CSTR_simulator = torch.jit.script(self.CSTR_simulator)
    

    def _initialize_scaler_torch(self):
        # scaler for rollouts
        self.observation_scaler_torch = TorchMinMaxScaler(feature_range=(-1, 1))
        self.observation_scaler_torch.fit(min_val = torch.tensor(self.env.observation_scaler.data_min_[:-1].reshape(1,-1),device=self.device), max_val = torch.tensor(self.env.observation_scaler.data_max_[:-1].reshape(1,-1),device=self.device))
        self.action_scaler_torch = TorchMinMaxScaler(feature_range=(-1, 1))
        self.action_scaler_torch.fit(min_val = torch.tensor(self.env.action_scaler.data_min_[:-1].reshape(1,-1),device=self.device), max_val = torch.tensor(self.env.action_scaler.data_max_[:-1].reshape(1,-1),device=self.device))
    

    def _setup_agent(self):
         # load agent
        self.agent = stable_baselines3.TD3.load(
            rf'{self.agent_dir}\best_model\best_model.zip',
            custom_objects={"buffer_size": 100000},
            device=self.device
        )
        self.agent = self.agent.policy
        self.agent = self.agent.to(self.device)
    

    def _setup_backup_policy(self):
        # load hyperparameters from json file
        with open(os.path.join(self.backup_dir, 'hyperparameters.json'), 'rb') as f:
                hyperparameters = json.load(f)
        
        # load model
        self.backup_policy = ARMPC_CSTR(input_size=hyperparameters['input_size'], hidden_size = hyperparameters['num_neurons'], num_hidden_layers = hyperparameters['num_hidden_layers'], layer_act = hyperparameters['activation_function']) # input size might needs to be adjusted
        self.backup_policy.load_state_dict(torch.load(os.path.join(self.backup_dir, 'best_model.pth'), weights_only = True, map_location = self.device))
        self.backup_policy = self.backup_policy.to(self.device)
        self.backup_policy.eval()


    def _check_constraint_violation_batch(self, states: torch.Tensor):
        """
        Optimized constraint violation checking with rounding to first decimal.
        """
        # Use only the first 4 dimensions and round to first decimal
        physical_states = torch.round(states[:, :4], decimals=1)
    
        # Direct comparison - self.lbx_tensor and self.ubx_tensor are already (1, 4)
        violations = (physical_states < self.lbx_tensor) | (physical_states > self.ubx_tensor)
    
        # Return True if any dimension violates
        return violations.any(dim=1)
    
    
    def _rollout_candidate_action(self, num_MC:int = None):
        # Alpha and beta combinations
        if num_MC is not None:
            alpha_min = beta_min = 1 - self.env.safety_parameter
            alpha_max = beta_max = 1 + self.env.safety_parameter

            alphas = np.random.uniform(alpha_min, alpha_max, size=num_MC)
            betas = np.random.uniform(beta_min, beta_max, size=num_MC)

            alpha_beta_combinations = [[alpha, beta] for alpha, beta in zip(alphas, betas)]

        else: 
            # combination of extreme values of alpha and beta
            alphas = [1 - self.env.safety_parameter, 1 + self.env.safety_parameter]
            betas = [1 - self.env.safety_parameter, 1 + self.env.safety_parameter]
            
            # Create all combinations at once
            alpha_beta_combinations = []
            for alpha in alphas:
                for beta in betas:
                    alpha_beta_combinations.append([alpha, beta])
            
        num_combinations = len(alpha_beta_combinations)
        
        # Batch initial states and actions for all combinations
        current_state = torch.tensor(self.env.state_phy, device=self.device).reshape(1, -1)
        current_state_batch = current_state.repeat(num_combinations, 1)
        
        backup_action = torch.tensor(self.action_candidate_num, device=self.device).reshape(1, -1)
        backup_action_batch = backup_action.repeat(num_combinations, 1)
        
        alpha_beta_batch = torch.tensor(alpha_beta_combinations, device=self.device, dtype=torch.float32)
        
        try:
            # First step for all combinations at once
            sub_state_batch = self.CSTR_simulator.make_step_batch(
                current_state_batch, backup_action_batch, alpha_beta_batch
            )
            
            for _ in range(self.n_horizon):
                # Check constraints for all combinations
                violations = self._check_constraint_violation_batch(sub_state_batch)
                
                # If any combination violates constraints, return False immediately
                if violations.any():
                    return False
                
                # Prepare batch observations for the neural network
                sub_state_np = sub_state_batch.cpu().numpy()
                backup_action_np = backup_action_batch.cpu().numpy()
                
                # Create safety parameter array for all combinations
                safety_params = np.full((num_combinations, 1), self.safety_parameter)

                # Combine all features for the batch
                combined_batch = np.concatenate([
                    sub_state_np,
                    backup_action_np,
                    safety_params
                ], axis=1)
                
                # Scale all observations at once
                observations_scaled = self.env.observation_scaler.transform(combined_batch)
                
                # Convert to torch tensor for the neural network
                observations_tensor = torch.tensor(
                    observations_scaled, 
                    device=self.device, 
                    dtype=torch.float32
                )

                if hasattr(self, 'backup_policy'):
                    # Get batch predictions from backup policy
                    with torch.no_grad():  
                        actions_scaled_tensor = self.backup_policy(observations_tensor[:,:-1])
                
                else:
                    # Get batch predictions from the agent 
                    with torch.no_grad():  
                        actions_scaled_tensor = self.agent(observations_tensor)
                

                # Convert back to numpy for rescaling
                actions_scaled_np = actions_scaled_tensor.cpu().numpy()
                actions_scaled_np = np.clip(actions_scaled_np, -1,1)

                
                # Rescale all actions at once
                backup_actions_np = self.env.action_scaler.inverse_transform(actions_scaled_np)
                backup_action_batch = torch.tensor(backup_actions_np, device=self.device)
                
                # Next step
                sub_state_batch = self.CSTR_simulator.make_step_batch(
                    sub_state_batch, backup_action_batch, alpha_beta_batch
                )
                
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # If all rollouts did not lead to any constraint violation
        return True

    def update_safety_parameter_in_obs(self, obs, safety_parameter):
        obs_sp = obs.copy()
        obs_sp[-1] = safety_parameter
        obs[-1] = self.env.observation_scaler.transform(obs_sp.reshape(1,-1)).reshape(-1,1)[-1]

        return obs

    def _backup_predict(self, observation):
        action_backup = self.backup_policy(torch.tensor(observation.reshape(1,-1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
        return np.clip(action_backup,-1,1)

    def shielded_closed_loop(self, num_eval: int = 100, step_sp: int = 0.05, num_MC:int = None, sampling:str = 'uniform'):
        # Initialize lists to store results for each episode
        results_data = []

        np.random.seed(123)
        for i in range(num_eval):     
            self.i = i
            print(rf'start evaluation {i}')
            obs, _ = self.env.reset(seed = i, sampling = sampling
                                    )
            terminated = False
            CVs = 0
            CV_num = 0
            acc_safety = 0
            shield_intervention = 0
            mpc_objective = 0
            pred_time = 0
            number_steps = 0

            while not terminated:
                self.action = None
                self.safety_parameter = 0

                start_time = time.perf_counter()
                while self.action is None:
                    
                    # robust shielding of models without safety parameter
                    if hasattr(self, 'backup_policy') and not self.sp:
                        
                        # Get candidate action from agent
                        action_candidate=self.agent(torch.tensor(obs[:-1].reshape(1,-1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                        action_candidate = np.clip(action_candidate,-1,1)
                        self.action_candidate_num = self.env.action_scaler.inverse_transform(action_candidate.reshape(1,-1))

                        if not self._rollout_candidate_action(num_MC=num_MC):
                            # use backup policy for alternative action
                            action_backup = self._backup_predict(obs[:-1])
                            self.action = action_backup
                            shield_intervention += 1
                       
                        else:
                            self.action = action_candidate
                    
                    # robust shielding of models with safety parameter, but without adaptation of safety parameter
                    elif hasattr(self, 'backup_policy') and self.sp and not self.adaptive:
                       
                        # Get action from agent
                        action_candidate=self.agent(torch.tensor(obs.reshape(1,-1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                        action_candidate = np.clip(action_candidate,-1,1)
                        self.action_candidate_num = self.env.action_scaler.inverse_transform(action_candidate.reshape(1,-1))

                        if not self._rollout_candidate_action(num_MC=num_MC):
                            action_backup = self._backup_predict(obs[:-1])
                            self.action = action_backup
                            shield_intervention += 1
                

                        else:
                            self.action = action_candidate
                    

                    # robust adaptive / adaptive shielding of models with safety paramter
                    else:
                        
                        # update safety parameter in observation
                        obs = self.update_safety_parameter_in_obs(obs, self.safety_parameter)
                        
                        # Get candidate action from agent
                        action_candidate=self.agent(torch.tensor(obs.reshape(1,-1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                        action_candidate = np.clip(action_candidate,-1,1)
                        self.action_candidate_num = self.env.action_scaler.inverse_transform(action_candidate.reshape(1,-1))
                        
                        # rollout of candidate action
                        safe_action = self._rollout_candidate_action(num_MC=num_MC)
                        
                        # candidate action not safe, increase safety parameter
                        if (not safe_action) and (self.safety_parameter <= self.max_sp - step_sp):
                            self.safety_parameter += step_sp      
                        
                        # candidate action safe and applied
                        elif safe_action and (self.safety_parameter <= self.max_sp):
                            self.action = action_candidate
                        
                        # candidate action not safe, but applied as no backup policy is present (for adaptive shielding)
                        elif not hasattr(self, 'backup_policy'):
                            self.action = action_candidate
                        
                        # apply backup policy
                        else:
                            action_backup = self._backup_predict(obs[:-1])
                            self.action = action_backup
                            shield_intervention += 1


                pred_time += time.perf_counter() - start_time
                
                # Apply action in env
                obs, _, terminated, _, info = self.env.step(self.action)
                number_steps += 1
                
                acc_safety += self.safety_parameter
                mpc_objective += info['reward_MPC']
                
                if self.env._check_constraint_violation(self.env.state_phy):
                    CVs += self.env.constraint_deviation(self.env.state_phy)
                    CV_num += 1
                
       
            # After episode is complete, store results for this episode
            episode_result = {
                'episode': i,
                'mpc_objective': mpc_objective,
                'constraint_violations_num': CV_num,
                'constraint_violations_d': CVs,
                'pred_time': (pred_time/number_steps)*1000,       # in ms
                'accumulated_safety': acc_safety,
                'safety_parameter_env':self.env.safety_parameter,
                'alpha': self.env.p[0],
                'beta': self.env.p[1]
            }
            if hasattr(self, 'backup_policy') or self.sp:
                episode_result['shield_intervention'] = shield_intervention

            # Add this episode's data to the results list
            results_data.append(episode_result)
        
        # Convert results list to DataFrame
        self.results_df = pd.DataFrame(results_data)
    

class shielded_deployment_multi_stage_mpc(shielded_deployment):
    def __init__(self, agent_dir: str = None, env: gym.Env = None, seed: int = 123, max_sp:float = 0.2, sp: bool = False, adaptive: bool = True, n_horizon: int = 20, **kwargs):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_dir = agent_dir
        self.env = env
        self.seed = seed
        self.max_sp = max_sp
        self.sp = sp
        self.adaptive = adaptive       # for evaluation with non-adaptive robust shielding of agents with safety parammeter 
   
        # precompute safety bounds according to the environment 
        self.lbx = self.env.lbx_cv[:4].reshape(1, 4)
        self.ubx =self.env.ubx_cv[:4].reshape(1, 4)

        self._setup_model_rmpc()

        self.n_horizon = n_horizon
        self.integration_setting = integration_setting()
        
        self._setup_agent()
    

    def _setup_model_rmpc(self):
        self.model_rmpc = CSTR_model()

    def _rmpc_predict(self, obs_num):
         # set initial condition for multi-stage mpc
        self.rmpc.x0 = obs_num[:4].copy()
        self.rmpc.u0 = obs_num[4:6].copy()
        self.rmpc.set_initial_guess()

        rmpc_action = self.rmpc.make_step(obs_num[:4].copy())
        return self.env.action_scaler.transform(rmpc_action.reshape(1,-1)).reshape(-1,1)

    
    def shielded_closed_loop(self, num_eval: int = 100, step_sp: int = 0.05, num_MC:int = None):

        # Initialize lists to store results for each episode
        results_data = []

        np.random.seed(123)
        for i in range(num_eval):     
            self.i = i
            print(rf'start evaluation {i}')
            obs, _ = self.env.reset(seed = i
                                    )
            
            terminated = False
            CVs = 0
            CV_num = 0
            acc_safety = 0
            shield_intervention = 0
            mpc_objective = 0
            pred_time = 0
            number_steps = 0

            while not terminated:
                self.action = None
                self.safety_parameter = 0

                # initialize mpc controller according to environment uncertainty
                self.rmpc = CSTR_msMPC(model = self.model_rmpc, x_ub = np.array([2.0, 2.0,140,140]), x_lb = np.array([0.1, 0.1, 100,100]),
                 u_ub= np.array([100,0]), u_lb = np.array([5,-8500]), alpha_var = np.array([1 - self.env.safety_parameter, 1 + self.env.safety_parameter]), 
                 beta_var= np.array([1 - self.env.safety_parameter, 1 + self.env.safety_parameter]))
                

                start_time = time.perf_counter()
                while self.action is None:
                    obs_num = self.env.observation_scaler.inverse_transform(obs.reshape(1,-1)).reshape(-1,1)

                    # robust shielding of models without safety parameter
                    if not self.sp:
                        
                        # Get candidate action from agent
                        action_candidate=self.agent(torch.tensor(obs[:-1].reshape(1,-1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                        action_candidate = np.clip(action_candidate,-1,1)
                        self.action_candidate_num = self.env.action_scaler.inverse_transform(action_candidate.reshape(1,-1))

                        if not self._rollout_candidate_action(num_MC=num_MC):
                            self.action = self._rmpc_predict(obs_num)
                            shield_intervention += 1
                        else:
                            self.action = action_candidate
                    
                    # robust shielding of models with safety parameter, but without adaptation of safety parameter
                    elif self.sp and not self.adaptive:
                       
                        # Get action from agent
                        action_candidate=self.agent(torch.tensor(obs.reshape(1,-1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                        action_candidate = np.clip(action_candidate,-1,1)
                        self.action_candidate_num = self.env.action_scaler.inverse_transform(action_candidate.reshape(1,-1))

                        if not self._rollout_candidate_action(num_MC=num_MC):
                            self.action = self._rmpc_predict(obs_num)
                            shield_intervention += 1

                        else:
                            self.action = action_candidate
                    

                    # adaptive robust shielding of models with safety paramter
                    else:
                        # update safety parameter in observation
                        obs = self.update_safety_parameter_in_obs(obs, self.safety_parameter)
                        
                        # Get candidate action from agent
                        action_candidate=self.agent(torch.tensor(obs.reshape(1,-1), dtype=torch.float32).to(self.device)).cpu().detach().numpy()
                        action_candidate = np.clip(action_candidate,-1,1)
                        self.action_candidate_num = self.env.action_scaler.inverse_transform(action_candidate.reshape(1,-1))
                        
                        # rollout of candidate action
                        safe_action = self._rollout_candidate_action(num_MC=num_MC)
                        
                        # candidate action not safe, increase safety parameter
                        if (not safe_action) and (self.safety_parameter <= self.max_sp - step_sp):
                            self.safety_parameter += step_sp      
                        
                        # candidate action safe and applied
                        elif safe_action and (self.safety_parameter <= self.max_sp):
                            self.action = action_candidate
                        
                        # apply backup policy
                        else:
                            self.action = self._rmpc_predict(obs_num)
                            shield_intervention += 1

                pred_time += time.perf_counter() - start_time
                
                # Apply action in env
                obs, _, terminated, _, info = self.env.step(self.action)
                number_steps += 1
                
                acc_safety += self.safety_parameter
                mpc_objective += info['reward_MPC']
                
                if self.env._check_constraint_violation(self.env.state_phy):
                    CVs += self.env.constraint_deviation(self.env.state_phy)
                    CV_num += 1
                      
            # After episode is complete, store results for this episode
            episode_result = {
                'episode': i,
                'mpc_objective': mpc_objective,
                'constraint_violations_num': CV_num,
                'constraint_violations_d': CVs,
                'pred_time': (pred_time/number_steps)*1000,       # in ms
                'accumulated_safety': acc_safety,
                'safety_parameter_env':self.env.safety_parameter,
                'shield_intervention': shield_intervention,
                'alpha': self.env.p[0],
                'beta': self.env.p[1]
            }

            # Add episode's data to the results list
            results_data.append(episode_result)
        
        # Convert results list to DataFrame
        self.results_df = pd.DataFrame(results_data)

    
    def _rollout_candidate_action(self, num_MC:int = None):
        # Alpha and beta combinations
        if num_MC is not None:
            alpha_min = beta_min = 1 - self.env.safety_parameter
            alpha_max = beta_max = 1 + self.env.safety_parameter

            alphas = np.random.uniform(alpha_min, alpha_max, size=num_MC)
            betas = np.random.uniform(beta_min, beta_max, size=num_MC)

            alpha_beta_combinations = [[alpha, beta] for alpha, beta in zip(alphas, betas)]

        else: 
            # combination of extreme values of alpha and beta
            alphas = [1 - self.env.safety_parameter, 1 + self.env.safety_parameter]
            betas = [1 - self.env.safety_parameter, 1 + self.env.safety_parameter]
            
            # Create all combinations at once
            alpha_beta_combinations = []
            for alpha in alphas:
                for beta in betas:
                    alpha_beta_combinations.append([alpha, beta])
        
        for alpha_beta in alpha_beta_combinations:

            current_state = self.env.state_phy.copy()
            backup_action = self.action_candidate_num.copy()

            # initialize simulator according to alpha beta combination
            simulator_alpha_beta = CSTR_simulator(model = self.model_rmpc,p = alpha_beta )
            simulator_alpha_beta.x0 = current_state.copy()

            # apply candidate action
            current_state = simulator_alpha_beta.make_step(backup_action.reshape(-1,1))
            
            violations = self.check_constraint_violation_single(current_state)
            
            # check whether applicytion of candidate action leads to constraint violation
            if violations.any():
                return False
            
            # check whether multi-stage mpc finds feasible solution
            obs_num = np.concatenate((current_state, self.action_candidate_num.reshape(-1,1)), axis=0)
            _ = self._rmpc_predict(obs_num)
            if not self.rmpc.solver_stats['success']:
                return False            
        
        return True
    

    def check_constraint_violation_single(self, state):

        # Use only the first 4 dimensions and round to first decimal
        physical_state = np.round(state[:4], decimals=1).reshape(1,-1)
        
        # Check violations
        violations = (physical_state < self.lbx) | (physical_state > self.ubx)
        
        # Return True if any dimension violates
        return np.any(violations)