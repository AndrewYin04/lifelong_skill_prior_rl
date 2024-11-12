import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SkillPriorNet(nn.Module):

    def __init__(self, input_dim, latent_dim=128, skill_dim=10):
        super(SkillPriorNet, self).__init__()
        # Six fully connected layers with ReLU activations
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, latent_dim)
        self.fc6 = nn.Linear(latent_dim, latent_dim)
        # Output layer for mean and log variance of the Gaussian skill prior
        self.mean = nn.Linear(latent_dim, skill_dim)
        self.logvar = nn.Linear(latent_dim, skill_dim)

    def forward(self, x):
        # Pass through the 6 layers with ReLU activation
        x = x[:,0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # Output mean and log variance for the skill prior
        mean = self.mean(x)
        logvar = self.logvar(x)
        # Sample from the Gaussian skill prior using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std  # z ~ N(mean, variance)
        return mean, logvar