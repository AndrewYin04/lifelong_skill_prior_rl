import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim=128, skill_dim=10):
        super(PolicyNetwork, self).__init__()
        # Same architecture as SkillPriorNet
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, latent_dim)
        self.fc6 = nn.Linear(latent_dim, latent_dim)
        self.mean = nn.Linear(latent_dim, skill_dim)
        self.logvar = nn.Linear(latent_dim, skill_dim)

    def forward(self, state):
        # Pass through fully connected layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # Output Gaussian parameters
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar