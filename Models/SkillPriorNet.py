import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

class SkillPriorNet(nn.Module):

    def __init__(self, input_dim, latent_dim=128, skill_dim=10):
        super(SkillPriorNet, self).__init__()

        # CNN for feature extraction
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Remove classification head to get feature embeddings (512-dim)

        # Six fully connected layers with ReLU activations
        self.fc1 = nn.Linear(512*2 + input_dim, latent_dim) # 512 (CNN output) * 2 (Two cnn outputs) + input_dim (low-dim data)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, latent_dim)
        self.fc6 = nn.Linear(latent_dim, latent_dim)
        # Output layer for mean and log variance of the Gaussian skill prior
        self.mean = nn.Linear(latent_dim, skill_dim)
        self.logvar = nn.Linear(latent_dim, skill_dim)

    def forward(self, agentview_rgb, eye_in_hand_rgb, low_dim_features):
        '''
        Args:
            rgb_image: Tensor of shape [batch_size, 3, H, W] (raw RGB image)
            low_dim_features: Tensor of shape [batch_size, input_dim] (low-dimensional data)
        '''
        # Extract image features using the CNN
        agentview_features = self.cnn(agentview_rgb) # Output: [batch_size, 512]
        eye_in_hand_features = self.cnn(eye_in_hand_rgb) # Output: [batch_size, 512]

        # Concatenate CNN features with low-dimensional features
        x = torch.cat([agentview_features, eye_in_hand_features, low_dim_features], dim=-1)  # Shape: [batch_size, 512 + input_dim]

        # Pass through the 6 layers with ReLU activation
        #x = x[:,0]
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

if __name__ == "__main__":
    input_dim = 21
    model = SkillPriorNet(input_dim)

    # Define example inputs
    batch_size = 8  # Number of samples in the batch
    img_height, img_width = 128, 128  # Height and width of the images
    agentview_rgb = torch.randn(batch_size, 3, img_height, img_width)  # Random agentview image
    eye_in_hand_rgb = torch.randn(batch_size, 3, img_height, img_width)  # Random eye-in-hand image
    low_dim_features = torch.randn(batch_size, input_dim)  # Random low-dimensional features

    # Forward pass through the model
    mean, logvar = model(agentview_rgb, eye_in_hand_rgb, low_dim_features)

    # Print the outputs
    print("Mean shape:", mean.shape)  # Should be [batch_size, skill_dim]
    print("Log variance shape:", logvar.shape)  # Should be [batch_size, skill_dim]