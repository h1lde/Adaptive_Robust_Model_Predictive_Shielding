import numpy as np
import gymnasium as gym
import torch
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from msMPC.do_mpc_msMPC import CSTR_model, CSTR_simulator, CSTR_msMPC


@dataclass
class integration_setting:
    dt : float = 0.005      # NOTE: in hours 0.005 h equals 18 s
    opts : dict = None


class CSTR_sp(gym.Env):           # for evaluating agent without shielding, new definition of safety parameter
    def __init__(self, seed: int = 48, fix_p_unc: bool = False, feas_IC: bool = False, penalty_weight: float = 5e3, max_sp = 0.2):

        self.seed: int = seed
        self.rng = np.random.default_rng(seed = seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fix_p_unc = fix_p_unc
        self.feas_IC = feas_IC
        self.penalty_weight = penalty_weight
        self.max_sp = max_sp

        # Initialize the time
        self.time: float = 0.0      # NOTE: In hours
        self.max_time: float = 0.5  # NOTE: In hours 0.25 h equals 15 min 

        # system setup
        self.integration_setting = integration_setting()
        self._setup_model()
        self.simulator = self._setup_simulator()
        self._setup_bounds()
        self._setup_scaling()

        # environment space
        self.action_space = gym.spaces.Box(low = -np.ones(self.lbu.shape), high = np.ones(self.ubu.shape), dtype = np.float64)
        self.observation_space = gym.spaces.Box(low = np.zeros(self.lbx.shape) - np.inf, high = np.ones(self.ubx.shape) + np.inf, dtype = np.float64)

    def _setup_model(self):
        self.model = CSTR_model()

    def _setup_simulator(self, p: list = [1.0, 1.0]):

        simulator = CSTR_simulator(model = self.model, p = p)

        return simulator

    def _setup_bounds(self):

        self.lbx = np.array([
            0.1, # C_a
            0.1, # C_b
            120.0, # T_R        # 120 for older models^important for scaling!
            120.0, # T_K        # 120 for older models^important for scaling!
            5.0, # F
            -8500.0, # Q_dot
            0.0 # safety parameter
            ]).reshape(-1,1)

        self.ubx = np.array([
            2.0, # C_a
            2.0, # C_b
            140.0, # T_R
            140.0, # T_K
            100.0, # F
            0, # Q_dot
            0.5 # safety parameter
            ]).reshape(-1,1)

        self.lbu = np.array([
            5.0,
            -8500.0
            ]).reshape(-1,1)

        self.ubu = np.array([
            100.0,
            0.0
            ]).reshape(-1,1)


        # bounds for initialization
        self.lbx_init = np.array([
            0.6, # C_a
            0.35, # C_b
            132, # T_R
            128.0, # T_K
            5.0, # F
            -8500, # Q_dot
            0.0
            ]).reshape(-1,1)

        self.ubx_init = np.array([
            1.0, # C_a
            0.55, # C_b
            136, # T_R              # usually 136.0
            132.0, # T_K
            100.0, # F
            0, # Q_dot
            self.max_sp # safety parameter
            ]).reshape(-1,1)

        self.lbx_cv = np.array([
            0.1, # C_a
            0.1, # C_b
            100.0, # T_R
            100.0, # T_K
            5.0, # F
            -8500.0, # Q_dot
            0.0 # safety parameter
            ]).reshape(-1,1)

        self.ubx_cv = np.array([
            2.0, # C_a
            2.0, # C_b
            140.0, # T_R
            140.0, # T_K
            100.0, # F
            0, # Q_dot
            0.5 # safety parameter
            ]).reshape(-1,1)

    def _setup_scaling(self):
        # Initialize MinMaxScaler for actions

        self.action_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.action_scaler.fit(np.vstack((self.lbu.reshape(1,-1), self.ubu.reshape(1,-1))))

        # Initialize MinMaxScaler for observations
        self.observation_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.observation_scaler.fit(np.vstack((self.lbx.reshape(1,-1), self.ubx.reshape(1,-1))))

        # Initialize MinMaxScaler for backup policy without safety parameter
        self.backup_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.backup_scaler.fit(np.vstack((self.lbx[:-1].reshape(1,-1), self.ubx[:-1].reshape(1,-1))))

    def _MPC_objective(self, state: np.array, action: np.array):
        
        self.R = np.array ([          
            1e-1,           
            1e-3            
            ]).reshape(-1,1)
        
        self.mpc_objective = float((state[1] - 0.6)**2 + np.sum((self.R*((action-state[4:6])**2))))

        return self.mpc_objective

    def _calculate_reward(self, state: np.array, action: np.array):

        reward = self._MPC_objective(state, action)

        penalty = self.constraint_deviation(state = state[:4])*self.penalty_weight
        reward += penalty

        return -float(reward / 10)


    def _check_constraint_violation(self, state: np.array):
        
        # states are assumed to be unscaled and only physical states included, this is (only??) used for evaluation therefore rounding, for training no rounding is considered
        if np.all(np.maximum(0,np.round(self.lbx_cv[:4] - state, 1)) == 0) and np.all(np.maximum(0, np.round(state - self.ubx_cv[:4],1)) == 0):
            return False
        else:
            return True # CV dected

    def constraint_deviation(self, state):

        # Calculate lower bound violations (positive values indicate violation)
        lower_bound_violations = np.maximum(0, self.lbx_cv[:4] - state)

        # Calculate upper bound violations (positive values indicate violation)
        upper_bound_violations = np.maximum(0, state - self.ubx_cv[:4])

        # Calculate total deviation
        total_deviation = np.sum(lower_bound_violations) + np.sum(upper_bound_violations)

        return total_deviation

    def _check_termination_truncation(self):
        termination = False
        truncation = False

        if self.time >= self.max_time:
            termination = True

        return termination, truncation


    def step(self, action: np.array):

        # rescaling of action
        self.action_num = self.action_scaler.inverse_transform(action.reshape(1,-1)).reshape(-1,1)

        # calculate reward
        reward = self._calculate_reward(self.observation_num, self.action_num)

        # calculate next observation
        self.state_phy = self.simulator.make_step(self.action_num)
        self.simulator.reset_history()
        self.observation_num = np.concatenate((self.state_phy, self.action_num, np.array([self.safety_parameter]).reshape(-1,1)), axis = 0)
        self.observation = self.observation_scaler.transform(self.observation_num.reshape(1,-1)).reshape(-1,1)

        # update time
        self.time += self.integration_setting.dt

        # check truncation and termination
        termination, truncation = self._check_termination_truncation()

        info = {"reward_MPC": self.mpc_objective}

        return self.observation, reward, termination, truncation, info
    
    def _setup_mpc(self, p: list = [1.0, 1.0]):
        # define scenario9 tree accroding to present uncertainty
        param = np.array([1-self.safety_parameter, 1+self.safety_parameter])

        # setup multi-stage MPC
        self.mpc = CSTR_msMPC(model = self.model, alpha_var = param, beta_var = param) 


    def reset(self, seed: int = None, safety_parameter = None, p: list = None, sampling:str = 'uniform'):

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed = seed)

    
        self.observation_num = self.rng.uniform(self.lbx_init, self.ubx_init)
        
        # beta distribution sampling of uncertainty instead of uniform sampling
        if sampling == 'beta':
            a, b = 5, 2     # beta distribution towards higher uncertainty
            self.safety_parameter = self.rng.beta(a, b)*(self.max_sp)

        if safety_parameter is not None:
            self.safety_parameter = safety_parameter
            self.observation_num[-1][0] = self.safety_parameter
        else:
            self.safety_parameter = self.observation_num[-1][0]    # random selection of safety parameter

        # initialize the simulator
        if p is not None:
            self.p = p
        elif self.fix_p_unc:            # for training 
            self.p = [self.rng.choice([1 - self.safety_parameter, 1 + self.safety_parameter]),
                      self.rng.choice([1 - self.safety_parameter, 1 + self.safety_parameter])]
        else:                           # for evaluation        
            self.p = [self.rng.uniform(1 - self.safety_parameter, 1 + self.safety_parameter),
                    self.rng.uniform(1 - self.safety_parameter, 1 + self.safety_parameter)]

        # sampling of feasible initial conditions only
        if self.feas_IC:   
            feasible = False
            self._setup_mpc(p = self.p)
            while not feasible:
                self.mpc.x0 = self.observation_num[0:4].reshape(-1,1)
                self.mpc.set_initial_guess()

                _ = self.mpc.make_step(self.observation_num[0:4].reshape(-1,1))

                # check whether solver found a solution
                solver_stats =  self.mpc.solver_stats
                if solver_stats['success']:
                    feasible = True
                else:
                    feasible = False
                    # sample new initial condition, keep safety paramter and uncertain parameters
                    self.observation_num = self.rng.uniform(self.lbx_init, self.ubx_init)
                    self.observation_num[-1][0] = self.safety_parameter


        # setup simulator with respecrt to parameters
        self.simulator= self._setup_simulator(p = self.p)     
        self.simulator.x0 = self.observation_num[0:4].reshape(-1,1)
        self.simulator.reset_history()

        self.state_phy = self.observation_num[0:4].reshape(-1,1)

        self.observation = self.observation_scaler.transform(self.observation_num.reshape(1,-1)).reshape(-1,1)

        self.action_num_old = self.observation_num[4:6].reshape(-1,1)

        # reset the time
        self.time = 0.0

        info = {}

        return self.observation, info



class CSTR_base(CSTR_sp):

    def __init__(self, seed: int = 48, penalty_weight: float = 5e3, max_sp: float = 0.2):

        self.seed: int = seed
        self.rng = np.random.default_rng(seed = seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.penalty_weight = penalty_weight
        self.max_sp = max_sp

        # Initialize the time
        self.time: float = 0.0      # NOTE: In hours
        self.max_time: float = 0.5  # NOTE: In hours 

        # system setup
        self.integration_setting = integration_setting()
        self._setup_model()
        self.simulator = self._setup_simulator()    
        self._setup_bounds() 
        self._setup_scaling()
       
        # environment space
        self.action_space = gym.spaces.Box(low = -np.ones(self.lbu.shape), high = np.ones(self.ubu.shape), dtype = np.float64)
        self.observation_space = gym.spaces.Box(low = np.zeros(self.lbx.shape) - np.inf, high = np.ones(self.ubx.shape) + np.inf, dtype = np.float64)


    def _setup_bounds(self):
        super()._setup_bounds()
        
        # remove safety parameter from bounds
        self.lbx = self.lbx[:-1] 
        self.ubx = self.ubx[:-1]
        self.lbx_init = self.lbx_init[:-1]
        self.ubx_init = self.ubx_init[:-1]
        self.lbx_cv = self.lbx_cv[:-1]
        self.ubx_cv = self.ubx_cv[:-1]

      
    def _setup_scaling(self):
        # Initialize MinMaxScaler for actions
        self.action_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.action_scaler.fit(np.vstack((self.lbu.reshape(1,-1), self.ubu.reshape(1,-1))))
        
        # Initialize MinMaxScaler for observations
        self.observation_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.observation_scaler.fit(np.vstack((self.lbx.reshape(1,-1), self.ubx.reshape(1,-1))))

        # Scaler for backup policy is the same as observation scaler as no safety parameter is considered
        self.backup_scaler = self.observation_scaler

    # exactly as in parent class?
    def step(self, action: np.array):
        
        # rescaling of action
        self.action_num = self.action_scaler.inverse_transform(action.reshape(1,-1)).reshape(-1,1)
        
        # calculate reward
        reward = self._calculate_reward(self.observation_num, self.action_num) 
      
        # calculate next observation
        self.state_phy = self.simulator.make_step(self.action_num)
        self.simulator.reset_history()
        self.observation_num = np.concatenate((self.state_phy, self.action_num), axis = 0)
        self.observation = self.observation_scaler.transform(self.observation_num.reshape(1,-1)).reshape(-1,1)

        # update time
        self.time += self.integration_setting.dt

        # check truncation and termination
        termination, truncation = self._check_termination_truncation()  

        info = {"reward_MPC": self.mpc_objective}

        return self.observation, reward, termination, truncation, info
    
    
    def reset(self, seed: int = None):

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed = seed)

        self.observation_num = self.rng.uniform(self.lbx_init, self.ubx_init)

        self.state_phy = self.observation_num[0:4].reshape(-1,1)  # physical states

        # initialize the simulator
        self.simulator = self._setup_simulator(p=[self.rng.uniform(1 - self.max_sp, 1 + self.max_sp), self.rng.uniform(1 - self.max_sp, 1 + self.max_sp)])     # parameter variance in environment
        self.simulator.x0 = self.observation_num[0:4].reshape(-1,1)
        self.simulator.reset_history()

        self.observation = self.observation_scaler.transform(self.observation_num.reshape(1,-1)).reshape(-1,1)
        
        self.action_num_old = self.observation_num[4:6].reshape(-1,1)

        # reset the time	
        self.time = 0.0

        info = {}

        return self.observation, info