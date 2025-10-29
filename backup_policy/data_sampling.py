import os
import pandas as pd
import gymnasium as gym
import numpy as np

from msMPC.do_mpc_msMPC import CSTR_model, CSTR_simulator, CSTR_msMPC

np.random.seed(48) 

def trajectory_sampling(env:gym.Env, num_traj: int, len_traj: int, save_dir: str, n_horizon: int = 20, n_robust: int = 1, t_step: float = 0.005, x_ub_t: np.array = np.array([1.9,1.9,138,138]), x_lb_t:np.array = np.array([0.2, 0.2, 102,102]), alpha_var:np.array = np.array([0.8, 1.2]), beta_var:np.array = np.array([0.8, 1.2])):
    model = CSTR_model()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Batch size for saving data
    batch_size = 1000  

    total_steps = 0
    
    # Temporary lists for batching
    x_batch = []
    y_batch = []
    feas_batch = []
    
    # Column definitions
    x_columns = ["C_a", "C_b", "T_R", "T_K", "F_old", "Q_dot_old"]
    y_columns = ["F", "Q_dot"]
    
    # Initialize CSV files with headers if they don't exist
    x_file = os.path.join(save_dir, "RMPC_raw_x_data.csv")
    y_file = os.path.join(save_dir, "RMPC_raw_y_data.csv")
    f_file = os.path.join(save_dir, "feas_check_raw_data.csv")
    
    # Create files with headers if they don't exist
    if not os.path.exists(x_file):
        pd.DataFrame(columns=x_columns).to_csv(x_file, index=False)
    if not os.path.exists(y_file):
        pd.DataFrame(columns=y_columns).to_csv(y_file, index=False)
    if not os.path.exists(f_file):
        pd.DataFrame(columns=['feas']).to_csv(f_file, index=False)
    
    def save_batch():
        """Helper function to save current batch to CSV files"""
        if x_batch:  # Only save if there's data
            x_df = pd.DataFrame(x_batch, columns=x_columns)
            y_df = pd.DataFrame(y_batch, columns=y_columns)
            f_df = pd.DataFrame(feas_batch, columns=['feas'])
            
            # Append to existing CSV files
            x_df.to_csv(x_file, mode='a', header=False, index=False)
            y_df.to_csv(y_file, mode='a', header=False, index=False)
            f_df.to_csv(f_file, mode='a', header=False, index=False)
            
            # Clear batches
            x_batch.clear()
            y_batch.clear()
            feas_batch.clear()

    for datapoint in range(num_traj):
        
        # sampling of initial state according to environment 
        _, _ = env.reset()                      

        # sampling only parameter with large deviation from nominal value
        p = [np.random.choice([np.random.uniform(alpha_var[0],alpha_var[0]+0.1), np.random.uniform(alpha_var[1]-0.1, alpha_var[1])]),
             np.random.choice([np.random.uniform(beta_var[0], beta_var[0]+0.1), np.random.uniform(beta_var[1]-0.1, beta_var[1])])]

        # setup simulator with sampled uncertain parameters
        simulator = CSTR_simulator(model, t_step, p)

        x0 = env.observation_num[0:4].reshape(-1, 1)
        u0 = env.observation_num[4:6].reshape(-1, 1)

        # set up multi-stage mpc with tightened constraints
        mpc = CSTR_msMPC(model = model, n_horizon = n_horizon, n_robust = n_robust,t_step = t_step, x_ub = x_ub_t, x_lb = x_lb_t, alpha_var = alpha_var, beta_var = beta_var)
        mpc.x0 = x0
        mpc.u0 = u0
        simulator.x0 = x0
        mpc.set_initial_guess()

        for step in range(len_traj):  
            x_batch.append(np.concatenate((x0.flatten(), u0.flatten())))
            
            try:
                u0 = mpc.make_step(x0)
                x0 = simulator.make_step(u0)
                y_batch.append(u0.flatten())
                feas_batch.append(1 if mpc.solver_stats['success'] else 0)
                simulator.reset_history()
                
            except RuntimeError as e:
                # Remove last element from all batches
                x_batch.pop()
                continue
            
            total_steps += 1
            
            # Save batch when it reaches the batch size
            if len(x_batch) >= batch_size:
                save_batch()
        
        # Progress reporting every 100 trajectories (optional)
        if datapoint % 100 == 0:
            print(f"Completed {datapoint} trajectories, {total_steps} total steps")
    
    # Save remaining data in the final batch
    save_batch()
    
    print(f"Trajectory sampling completed. Total steps: {total_steps}")


# function to omit unfeasible points sampled
def drop_infeasibles(x_data: pd.DataFrame, y_data: pd.DataFrame, feas_data:pd.DataFrame):
    mask = feas_data.iloc[:, 0] == 1

    # Gefilterte DataFrames erstellen
    x_data_red = x_data[mask].reset_index(drop=True)
    y_data_red = y_data[mask].reset_index(drop=True)

    return x_data_red, y_data_red


# function for drop duplicates
def drop_duplicates(x_data: pd.DataFrame, y_data: pd.DataFrame, n_dec: int):

    # scale data (also other scalers can be used)
    x_max = x_data.max(axis = 0)
    x_min = x_data.min(axis = 0)

    loc_x_data = (x_data - x_min) / (x_max - x_min)     # scale data and save as copy

    # round data to n_dec decimal values
    loc_x_data = loc_x_data.round(n_dec)                # round scaled values
    loc_x_data = loc_x_data.drop_duplicates()           # drop the duplicates

    # extract relevant data from copy (no rounding errors etc.)
    x_data = x_data.iloc[loc_x_data.index]
    y_data = y_data.iloc[loc_x_data.index]
    
    return x_data, y_data


# scale data according to environment
def scale_data(env: gym.Env, x_data: pd.DataFrame, y_data: pd.DataFrame):
    x_scaler = env.backup_scaler  # if safety parameter is not considered
    y_scaler=env.action_scaler

    x_data_scaled = x_scaler.transform(x_data.values)
    y_data_scaled = y_scaler.transform(y_data.values)

    return x_data_scaled, y_data_scaled

