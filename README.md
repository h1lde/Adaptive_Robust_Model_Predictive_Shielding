# Adaptive_Robust_Model_Predictive_Shielding

This repository contains the implementation of the methods presented in the paper: 
>**Safe Reinforcement Learning via Adaptive Robust Model Predictive Shielding**\
>Hilde Gerold and Sergio Lucia

## Overview

This project demonstrates a novel approach for safe deployment of learning-based controller via Adaptive Robust Model Predicitve Shielding. This repository includes:
1. **Approximate multi-stage MPC**: Construction of an approximate multi-stage NMPC  as a backup policy for model predicitve shielding, including data-sampling and neural network training.
2. **Augmented RL Training**: RL learning for an augmented observation space, which is extended with a safety parameter which can be iteratively adapted during deployment phase.
3. **Adaptive Robust Shielding Approaches**: Implementation of several robust model predicitve shielding approaches inlcuding the approximate multi-stage NMPC as a backup policy and the adaptive safety parameter.

## Repository Structure
```
├── backup_policy                       # Approximate Multi-stage MPC construction  
│   ├── auxiliary_functions_NN.py       # Neural network model and scaler   
│   ├── data_sampling.py                # Trajectory-based sampling
│   └── neural_network_training.py      # Neural network training function
├── msMPC                               # Multi-stage Model Predicitve Control implementation
│   └── do_mpc_msMPC.py                 # Multi-stage MPC using do-mpc
├── policy_deployment                   # Deployment of learning-based policies
│   ├── MPS.py                          # Model Predicitve Shielding approaches
│   ├── only_RL.py                      # Deployment without shield
│   └── torch_model_CSTR.py             # Torch implementation of CSTR model
├── RL_policy                           # RL training 
│   ├── auxiliary_functions_RL.py       # Scheduling function for RL
│   ├── RL_environments.py              # Environment with and without augmented observation space
│   └── RL_training.py                  # RL training script
├── LICENSE.txt                         # License file
├── README.md                           # This README file
└── Results_Adaptive_Robust_MPS.ipynb   # Jupyter Notebook to illustrate workflow
```


## Key Components

### Jupyter Notebook
The jupyter notebook `Results_Adaptive_Robust_MPS.ipynb` illustrates the workflow for reproducing the results presented in the work `Safe Reinforcement Learning via Adaptive Robust Model Predictive Shielding`. It is structured according to the Sections in the work and demonstrates the usage of the different python files and lists the applied hyperparamters and settings. \
**Note**: This is not the most efficient implementation for reproducing the results and we recommend parallelizing the RL training on the CPU and neural network training on the GPU. For the shielded evaluations we have limited the demonstration to a single model, which have to be extended if more models are to be evaluated. If the code python files are directly executed the path references might need to be adjusted as they are currently defined with respect to the jupyter notebook. 

### Model Predicitve Shielding Approaches
The python file `MPS.py` contains the different shielding approaches, which are investigated in the work. Depending on the inputs for the class `shielded_deployment`, **robust MPS**, **adaptive MPS** and **adaptive robust MPS** are implemented. In the jupyter notebook, we exemplarly deploy all shielding approaches for RL policies with and without safety parameter. 

## Usage


### Prerequisites

This project requires Python with the following packages:
- numpy
- pandas
- do-mpc
- Stable Baselines3
- gymnasium
- torch
- scikit-learn
- matplotlib
- jupyter
- casadi


## License

See the LICENSE.txt file for license rights and limitations.
