from backup_policy.auxiliary_functions_NN import ARMPC_CSTR
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import copy
import random
import torch.nn.functional as F

# seed
seed=48
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def train_ARMPC(X_data: np.array, Y_data: np.array, hyperparams: dict):
    
    # hyperparameter from dictionary
    num_neurons =  hyperparams['num_neurons']
    num_hidden_layers =  hyperparams['num_hidden_layers'] 
    learning_rate =  hyperparams['learning_rate']
    activation_function =  hyperparams['activation_function']
    batch_size =  hyperparams['batch_size']   
    regularization_param =  hyperparams['regularization_param']
    num_epochs =  hyperparams['num_epochs']
    early_stopping_ref = hyperparams['early_stopping_ref']
    save_path = hyperparams['save_dir'] + rf'\grid{hyperparams["number"]}'
    
    os.makedirs(save_path, exist_ok=True)
    
    # Get the specified GPU if it's in the hyperparams (for parallel GPU training)
    if 'gpu_id' in hyperparams and torch.cuda.is_available():
        device = torch.device(f"cuda:{hyperparams['gpu_id']}")
    else:
        # Otherwise, use any available GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print which device is being used for this training run
    print(f"Training model {hyperparams.get('number', 'unknown')} on {device}")
    
    # intialize the model
    model = ARMPC_CSTR(input_size=hyperparams['input_size'], hidden_size=num_neurons, num_hidden_layers=num_hidden_layers, layer_act=activation_function).to(device)
    
    # initialize loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_param)

    # Convert data to PyTorch tensors
    X_data = torch.tensor(X_data, dtype=torch.float32).to(device) 
    Y_data = torch.tensor(Y_data, dtype=torch.float32).to(device)

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.1765, random_state=seed)
    dataset_train = TensorDataset(X_train, Y_train)

    # intialize training values
    train_loss = []
    val_loss = []
    best_val_loss = np.inf
    best_model_params = None
    early_stopping = early_stopping_ref

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0

        for i, (X_batch, Y_batch) in enumerate(DataLoader(dataset_train, batch_size=batch_size, shuffle=True)):
            
            # Move batches to GPU
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss.append(train_loss_epoch / len(dataset_train))

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_val)
            val_loss_epoch = criterion(Y_pred, Y_val).item()
            val_loss.append(val_loss_epoch)

            # Save the best model
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_loss = train_loss[-1]
                best_model_params = copy.deepcopy(model.state_dict())
                early_stopping = early_stopping_ref
            else: 
                early_stopping -= 1

                if early_stopping == 0:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]}, Val Loss: {val_loss[-1]}')

    
    # save the best model 
    torch.save(best_model_params, rf'{save_path}\best_model.pth')

    # save training data
    training_data = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss, 'best_loss': best_loss})
    training_data.to_excel(rf'{save_path}\training_data.xlsx', index=False)

    # Save hyperparameters
    hyper_to_save = hyperparams.copy()
    if 'X_data' in hyper_to_save:
        del hyper_to_save['X_data']
    if 'Y_data' in hyper_to_save:
        del hyper_to_save['Y_data']
    if 'save_dir' in hyper_to_save:
        del hyper_to_save['save_dir']
    if 'gpu_id' in hyper_to_save:
        del hyper_to_save['gpu_id']
    
    with open(rf'{save_path}\hyperparameters.json', 'w') as f:
        json.dump(hyper_to_save, f)

    return best_loss, best_val_loss


def hyperparameter_list(X_train: np.array, Y_train: np.array, save_dir: str):
    
    # define hyperparameters
    num_neurons = [
                  32, 64
                   ]
    num_hidden_layers = [2,
                          3, 4
                         ]
    learning_rate = [5e-5, 1e-4
                     ]
    activation_function = ['relu'
                           ]
    batch_size = [256, 128
                  ]
    regularization_param = [5e-5]
    epochs = 12000
    early_stopping = 100
    number_grid = 1

    # grid for hyperparameter tuning
    hyperparameter_list = []
    for num_neuron in num_neurons:
        for num_hidden_layer in num_hidden_layers:
            for lr in learning_rate:
                for act_func in activation_function:
                    for bs in batch_size:
                        for reg_param in regularization_param:
                            hyperparameter_list.append({
                                'X_data': X_train,
                                'Y_data': Y_train,
                                'save_dir': save_dir,
                                'num_neurons': num_neuron,
                                'num_hidden_layers': num_hidden_layer,
                                'learning_rate': lr,
                                'activation_function': act_func,
                                'batch_size': bs,
                                'regularization_param': reg_param,
                                'num_epochs': epochs,
                                'early_stopping_ref': early_stopping,
                                'input_size': 6,                    
                                'number' : number_grid
                                })
                            number_grid += 1
    return hyperparameter_list


def eval_on_test_data(X_test, Y_test, save_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'r') as f:
        hyperparameters = json.load(f)
    backup_policy = ARMPC_CSTR(input_size = hyperparameters['input_size'], hidden_size = hyperparameters['num_neurons'], num_hidden_layers = hyperparameters['num_hidden_layers'], layer_act = hyperparameters['activation_function'])
    backup_policy.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=True, map_location = device ))
    backup_policy.eval()

    # Convert data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)  
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # Evaluate the model on the test set
    backup_policy.eval()
    with torch.no_grad():
        Y_pred = backup_policy(X_test)
        loss = F.mse_loss(Y_pred, Y_test).item()

    return loss