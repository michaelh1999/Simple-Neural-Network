import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Create a model class that inherits nn.Module
class Model(nn.Module):
    # Input Layer (4 features of the flower) -> 
    # Hidden Layer1 (n) -> 
    # H2 (n) -> 
    # output (3 classes of iris flowers)
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
# Pick a random seed for randomization
torch.manual_seed(41)

# Create an instance of a model
model = Model()