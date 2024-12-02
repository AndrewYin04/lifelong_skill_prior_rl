"""
this is the training loop

TODO
    - turn the cnn into something inside the SkillPrior model, rather than in this training loop, and make sure that the cnn gets trained as well 
- ask Dr. A to review our code cuz loss iss decreasing but very slightly (for some reason alr starts off super low...)
- ask Dr. A about RL stuff (what to go about etc)
"""

from Models.SkillEmbeddingPrior import SkillEmbeddingAndPrior
from Models.SkillPriorNet import SkillPriorNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from libero.lifelong.datasets import get_dataset
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from colorama import Fore, Style
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, Resize

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Using VAE model defined above...
dataset_path = 'libero/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5'
obs_modality = {
    'low_dim': ['ee_ori', 'ee_pos', 'ee_states', 'gripper_states', 'joint_states'],
    'rgb': ['agentview_rgb', 'eye_in_hand_rgb'],  # High-dimensional modalities
}
seq_len = 50  # You might have to alter this+ 
frame_stack = 1
batch_size = 16
num_actions = 7
latent_dim = 10
num_epochs = 100 # Modify as needed

# Get the dataset and DataLoader
dataset, shape_meta = get_dataset(
    dataset_path=dataset_path,
    obs_modality=obs_modality,  # Includes only low-dimensional data
    initialize_obs_utils=True,
    seq_len=seq_len,
    frame_stack=frame_stack,
    filter_key=None,
    hdf5_cache_mode="low_dim",
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Initialize the model, loss function, and optimizer
input_dim = num_actions
prior_dim = 21  # Adjusted since we no longer have image features
model = SkillEmbeddingAndPrior(input_dim, prior_dim).to(device)
reconstruction_loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

prior_loss_array = []
total_loss_array = []

image_transforms = Compose([
    Resize((128, 128), antialias=True),  # Resize to match model input
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss_epoch = 0
    reconstruction_loss_epoch = 0
    regularization_loss_epoch = 0
    prior_loss_epoch = 0

    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        # Extract low-dimensional data
        low_dim_data = torch.cat([
            data["obs"]["ee_ori"],
            data["obs"]["ee_pos"],
            data["obs"]["ee_states"],
            data["obs"]["gripper_states"],
            data["obs"]["joint_states"]
        ], dim=-1).to(device)

        # Combine low-dimensional data (use only the first state in the sequence)
        first_state = low_dim_data[:, 0, :]  # Shape: [batch_size, total_low_dim_features]
        states_input = first_state

        rgb_image = []
        for img_key in obs_modality['rgb']:
            img_seq = data["obs"][img_key]  # Shape: [batch_size, seq_len, C, H, W]
            first_frame = img_seq[:, 0]  # Take the first frame, shape: [batch_size, C, H, W]
            # image = first_frame.permute(0, 3, 1, 2)  # Convert to [batch_size, C, H, W]
            rgb_image.append(image_transforms(first_frame).to(device).float())

        actions_input = data['actions']  # Shape: [batch_size, seq_len, num_actions]
        actions_input = actions_input.to(device).float()

        # Pass data through the model
        reconstructed_x, mean, logvar, prior_mean, prior_logvar = model(actions_input, rgb_image[0], rgb_image[1], states_input) # agentview_rgb, eye_in_hand_rgb, low_dim_features

        # Compute reconstruction loss
        x_flat = actions_input.view(actions_input.size(0), -1)
        reconstructed_x_flat = reconstructed_x.view(reconstructed_x.size(0), -1)
        reconstruction_loss = reconstruction_loss_fn(reconstructed_x_flat, x_flat)
        reconstruction_loss_epoch += reconstruction_loss.item()

        # Encoder KL divergence
        q_z = Normal(mean, torch.exp(0.5 * logvar))
        p_z = Normal(torch.zeros_like(mean), torch.ones_like(torch.exp(0.5 * logvar)))
        encoder_kl_loss = kl_divergence(q_z, p_z).mean()
        regularization_loss_epoch += encoder_kl_loss.item()

        # Detach mean and logvar for prior loss computation
        mean_detached = mean.detach()
        logvar_detached = logvar.detach()

        # Skill prior KL divergence
        p_z_prior = Normal(prior_mean, torch.exp(0.5 * prior_logvar))
        q_z_detached = Normal(mean_detached, torch.exp(0.5 * logvar_detached))
        skill_prior_kl_loss = kl_divergence(q_z_detached, p_z_prior).mean()
        prior_loss_epoch += skill_prior_kl_loss.item()

        # Total loss
        total_loss = reconstruction_loss + 0.01 * encoder_kl_loss + skill_prior_kl_loss
        total_loss.backward()
        optimizer.step()
        total_loss_epoch += total_loss.item()

    avg_loss_epoch = total_loss_epoch / len(dataloader)
    avg_reconstruction_loss = reconstruction_loss_epoch / len(dataloader)
    avg_regularization_loss = regularization_loss_epoch / len(dataloader)
    avg_prior_loss = prior_loss_epoch / len(dataloader)

    total_loss_array.append(avg_loss_epoch)
    prior_loss_array.append(avg_prior_loss)


    # print(f"Epoch [{epoch + 1}/{num_epochs}], Average Total Loss: {avg_loss_epoch:.4f}")
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Average Regularization Loss: {avg_regularization_loss:.4f}")
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Average Prior Loss: {avg_prior_loss:.4f}")
    print(f"{Fore.YELLOW}Epoch [{epoch + 1}/{num_epochs}], Average Total Loss: {avg_loss_epoch:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Epoch [{epoch + 1}/{num_epochs}], Average Reconstruction Loss: {avg_reconstruction_loss:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Epoch [{epoch + 1}/{num_epochs}], Average Regularization Loss: {avg_regularization_loss:.4f}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Epoch [{epoch + 1}/{num_epochs}], Average Prior Loss: {avg_prior_loss:.4f}{Style.RESET_ALL}")


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), total_loss_array, label='Total Loss', marker='o')
plt.plot(range(1, 101), prior_loss_array, label='Prior Loss', marker='x')

# Labels and Title
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.title('Loss Over Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# Save the plot as an image file
plt.savefig('loss_plot.png')  # Save as PNG (or 'loss_plot.pdf' for PDF)
