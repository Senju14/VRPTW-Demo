import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def load_dqn_model(model_path, state_dim, action_dim):
    from safetensors.torch import load_file
    
    model = QNetwork(state_dim, action_dim)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model
