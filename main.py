from Models.SkillEmbeddingPrior import SkillEmbeddingAndPrior
from Models.SkillPriorNet import SkillPriorNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from libero.lifelong.datasets import get_dataset
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence



# using vae model defined above...
dataset_path = 'libero/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5' 
obs_modality = {
    'low_dim': ['agentview_rgb', 'ee_ori', 'ee_pos', 'ee_states', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
}
seq_len = 50 # todo: might have to alter this 
frame_stack = 1
batch_size = 16
num_actions = 7
latent_dim = 10
num_epochs = 100 # can modify as needed

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# breakpoint()

# Get the dataset and DataLoader
dataset, shape_meta = get_dataset(
    dataset_path=dataset_path,
    obs_modality=obs_modality, # what the heck is obs_modailty doing in this get_dataset function
    initialize_obs_utils=True,
    seq_len=seq_len,
    frame_stack=frame_stack,
    filter_key=None,
    hdf5_cache_mode="low_dim",
)
# breakpoint()
# print(dataset)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
# input_dim = seq_len * num_actions  # Ask nicholas what actually is input_dim lol

# breakpoint()
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_reserved())
# breakpoint()
# torch.cuda.empty_cache()

input_dim = num_actions
model = SkillEmbeddingAndPrior(input_dim).to(device)
reconstruction_loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss_epoch = 0
    reconstruction_loss_epoch = 0
    regularization_loss_epoch = 0
    prior_loss = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        # print(data['actions'].shape)
        # print(data["obs"]["ee_ori"].shape)
        # Assuming 'actions' is the key for input data
        x = data['actions']  # Shape: [batch_size, seq_len, num_actions]
        
        # breakpoint()
        x = x.to(device).float()

        reconstructed_x, mean, logvar, prior_mean, prior_logvar = model(x)
        # Check prior_logvar for NaNs or Infs
        if torch.isnan(prior_logvar).any():
            print('prior_logvar contains NaNs')
        if torch.isinf(prior_logvar).any():
            print('prior_logvar contains Infs')
        x_flat = x.view(x.size(0), -1)
        reconstructed_x_flat = reconstructed_x.view(reconstructed_x.size(0), -1) 
        reconstruction_loss = reconstruction_loss_fn(reconstructed_x_flat, x_flat)
        reconstruction_loss_epoch += reconstruction_loss.item()
        # Encoder KL divergence
        q_z = Normal(mean, torch.exp(0.5 * logvar))
        p_z = Normal(torch.zeros_like(mean), torch.ones_like(torch.exp(0.5 * logvar)))
        encoder_kl_loss = kl_divergence(q_z, p_z).mean()
        regularization_loss_epoch += encoder_kl_loss.item()
        # Detach mean and logvar
        mean_detached = mean.detach()
        logvar_detached = logvar.detach()

        # Skill prior KL divergence
        p_z_prior = Normal(mean, torch.exp(0.5 * logvar))
        q_z_detached = Normal(mean_detached, torch.exp(0.5 * logvar_detached))
        skill_prior_kl_loss = kl_divergence(q_z_detached, p_z_prior).mean()
        prior_loss += skill_prior_kl_loss.item()
        
        # Total loss
        total_loss = reconstruction_loss - 0.0001 * encoder_kl_loss + skill_prior_kl_loss
        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()

    avg_loss_epoch = total_loss_epoch / len(dataloader)
    avg_loss_reconstruction = reconstruction_loss_epoch / len(dataloader)
    avg_loss_regularization = regularization_loss_epoch / len(dataloader)
    avg_loss_prior = prior_loss / len(dataloader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss_epoch:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Reconstruction Loss: {avg_loss_reconstruction:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average regularization Loss: {avg_loss_regularization:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average prior Loss: {avg_loss_prior:.4f}")

        