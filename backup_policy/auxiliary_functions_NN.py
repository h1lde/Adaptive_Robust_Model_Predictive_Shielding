import torch.nn as nn
import torch as th
import numpy as np

class ARMPC_CSTR(nn.Module):
    def __init__(self, input_size=6,  output_size=2, hidden_size=12, num_hidden_layers=2, layer_act='relu'):
        super(ARMPC_CSTR, self).__init__()
        
        # Initialize the layers list with the first input layer
        if layer_act == 'tanh':
            layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        elif layer_act == 'relu':
            layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        
        # Add the hidden layers based on num_hidden_layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if layer_act == 'tanh':
                layers.append(nn.Tanh())
            elif layer_act == 'relu':
                layers.append(nn.ReLU())
        
        # Add the output layer with linear activation
        layers.append(nn.Linear(hidden_size, output_size))

        # Use nn.Sequential to create the model
        self.mu = nn.Sequential(*layers)

    def forward(self, x):
        return self.mu(x)
    


class TorchMinMaxScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None
        self.scale_ = None
        self.min_ = None
        # use cuda if available
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def fit(self, tensor=None, min_val=None, max_val=None):
        if tensor is not None:
            if min_val is None:
                self.min_val, _ = th.min(tensor, dim=0)
            else:
                self.min_val = min_val.clone().detach() if isinstance(min_val, th.Tensor) else th.tensor(min_val, dtype=tensor.dtype, device=tensor.device)

            if max_val is None:
                self.max_val, _ = th.max(tensor, dim=0)
            else:
                self.max_val = max_val.clone().detach() if isinstance(max_val, th.Tensor) else th.tensor(max_val, dtype=tensor.dtype, device=tensor.device)
        else:
            if min_val is not None and max_val is not None:
                self.min_val = min_val.clone().detach() if isinstance(min_val, th.Tensor) else th.tensor(min_val, device=self.device)
                self.max_val = max_val.clone().detach() if isinstance(max_val, th.Tensor) else th.tensor(max_val, device=self.device)
            else:
                raise ValueError("When no tensor is provided, both min_val and max_val must be provided")

        data_range = self.max_val - self.min_val
        self.scale_ = ((self.feature_range[1] - self.feature_range[0]) / data_range).to(self.device)
        self.min_ = (self.feature_range[0] - self.min_val * self.scale_).to(self.device)

    def transform(self, tensor):
        tensor = tensor.to(self.device)
        return tensor * self.scale_ + self.min_

    def inverse_transform(self, tensor):
        return (tensor - self.min_) / self.scale_

    def fit_transform(self, tensor=None, min_val=None, max_val=None):
        self.fit(tensor=tensor, min_val=min_val, max_val=max_val)
        return self.transform(tensor)

    def get_params(self):
        return {'min_val': self.min_val, 'max_val': self.max_val, 'scale': self.scale_, 'min': self.min_}
    

def compute_N_prob(epsilon, r, delta):
    term1 = r - 1
    term2 = np.log(1 / delta)
    term3 = np.sqrt(2 * (r - 1) * np.log(1 / delta))
    N = (1 / epsilon) * (term1 + term2 + term3)
    return int(np.ceil(N))