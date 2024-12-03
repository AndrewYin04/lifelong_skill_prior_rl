import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, skill_dim=10, latent_dim=128):
        super(CriticNetwork, self).__init__()
        # Same 6-layer architecture as SkillPriorNet
        self.fc1 = nn.Linear(state_dim + skill_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, latent_dim)
        self.fc6 = nn.Linear(latent_dim, latent_dim)
        # Output a single Q-value
        self.q_value = nn.Linear(latent_dim, 1)

    def forward(self, state, skill):
        # Concatenate state and skill
        x = torch.cat([state, skill], dim=-1)
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # Output Q-value
        q = self.q_value(x)
        return q
