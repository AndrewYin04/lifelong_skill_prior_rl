from Models.SkillEmbeddingPrior import SkillEmbeddingAndPrior
from Models.SkillPriorNet import SkillPriorNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from libero.lifelong.datasets import get_dataset

# using vae model defined above...
dataset_path = 'libero/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5' 
obs_modality = {
    'low_dim': ['agentview_rgb', 'ee_ori', 'ee_pos', 'ee_states', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
}
seq_len = 50
frame_stack = 1
batch_size = 16
num_actions = 7
latent_dim = 10
num_epochs = 100 # can modify as needed

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the dataset and DataLoader
dataset, shape_meta = get_dataset(
    dataset_path=dataset_path,
    obs_modality=obs_modality,
    initialize_obs_utils=True,
    seq_len=seq_len,
    frame_stack=frame_stack,
    filter_key=None,
    hdf5_cache_mode="low_dim",
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
input_dim = seq_len * num_actions  # Ask nicholas what actually is input_dim lol
# breakpoint()
model = SkillEmbeddingAndPrior(input_dim).to(device)
reconstruction_loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss_epoch = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        # Assuming 'actions' is the key for input data
        x = data['actions']  # Shape: [batch_size, seq_len, num_actions]
        x = x.to(device).float()

        reconstructed_x, mean, logvar, prior_mean, prior_logvar = model(x)
        x_flat = x.view(x.size(0), -1)
        reconstruction_loss = reconstruction_loss_fn(reconstructed_x, x_flat)

        # Encoder KL divergence
        q_z = Normal(mean, torch.exp(0.5 * logvar))
        p_z = Normal(torch.zeros_like(mean), torch.ones_like(mean))
        encoder_kl_loss = kl_divergence(q_z, p_z).mean()

        # Detach mean and logvar
        mean_detached = mean.detach()
        logvar_detached = logvar.detach()

        # Skill prior KL divergence
        p_z_prior = Normal(prior_mean, torch.exp(0.5 * prior_logvar))
        q_z_detached = Normal(mean_detached, torch.exp(0.5 * logvar_detached))
        skill_prior_kl_loss = kl_divergence(q_z_detached, p_z_prior).mean()

        # Total loss
        total_loss = reconstruction_loss - 0.01 * encoder_kl_loss + skill_prior_kl_loss
        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
              f"Loss: {total_loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f}, "
              f"Encoder KL Loss: {encoder_kl_loss.item():.4f}, Skill Prior KL Loss: {skill_prior_kl_loss.item():.4f}")

    avg_loss_epoch = total_loss_epoch / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss_epoch:.4f}")