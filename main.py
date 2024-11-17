"""
this is the training loop

TODO
- turn the cnn into something inside the SkillPrior model, rather than in this training loop, and make sure that the cnn gets trained as well 
- ask Dr. A to review our code on the cnn; should it get trained in the first place?
- how can we make it a variable sequence length? right now it's fixed at 50, but the libero datasets seem to have sequence lenghts of 100+
- i'm p sure the prior_mean, prior_logvar, mean, and logvar are not used correctly when caluclating losses for p and q; so i changed it to what i think is right.
- for some reason, the losses are increasing for prior loss and encoder kl loss
"""

from Models.SkillEmbeddingPrior import SkillEmbeddingAndPrior
from Models.SkillPriorNet import SkillPriorNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from libero.lifelong.datasets import get_dataset
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.models import ResNet18_Weights
import os

pid = os.getpid()
print(f"My process PID: {pid}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for image data
image_transforms = Compose([
    # ToTensor(),  # Convert image to tensor (C, H, W)
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    Resize((128, 128), antialias=True),  # Resize images to a fixed size
])

# Load pre-trained CNN for feature extraction
cnn_model = resnet18(weights=ResNet18_Weights.DEFAULT)
cnn_model.fc = nn.Identity()  # Remove classification layer to use as feature extractor
cnn_model = cnn_model.to(device)
cnn_model.eval()  # Set to evaluation mode

# using vae model defined above...
dataset_path = 'libero/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5' 
obs_modality = {
    'low_dim': ['ee_ori', 'ee_pos', 'ee_states', 'gripper_states', 'joint_states'],
    'rgb': ['agentview_rgb', 'eye_in_hand_rgb'],  # High-dimensional modalities
}
seq_len = 50 # TODO: might have to alter this 
frame_stack = 1
batch_size = 16
num_actions = 7
latent_dim = 10
num_epochs = 100 # can modify as needed

# Get the dataset and DataLoader
dataset, shape_meta = get_dataset(
    dataset_path=dataset_path,
    obs_modality=obs_modality, # Includes both low and high-dimensional data
    initialize_obs_utils=True,
    seq_len=seq_len,
    frame_stack=frame_stack,
    filter_key=None,
    hdf5_cache_mode="low_dim",
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Initialize the model, loss function, and optimizer
# input_dim = seq_len * num_actions  # Ask nicholas what actually is input_dim lol

# breakpoint()
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_reserved())
# breakpoint()
# torch.cuda.empty_cache()

input_dim = num_actions
prior_dim = 21 + 1024 
model = SkillEmbeddingAndPrior(input_dim, prior_dim).to(device)
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

        # Extract low-dimensional data
        low_dim_data = torch.cat([
            data["obs"]["ee_ori"], 
            data["obs"]["ee_pos"], 
            data["obs"]["ee_states"], 
            data["obs"]["gripper_states"], 
            data["obs"]["joint_states"]
        ], dim=-1).to(device)

        # Extract and process high-dimensional image data
        rgb_features = []
        for img_key in obs_modality['rgb']:
            img_seq = data["obs"][img_key]  # Shape: [batch_size, seq_len, H, W, C]
            batch_features = []
            for img_batch in img_seq:  # Process each batch
                batch_img_features = []
                for img in img_batch:  # Process each image
                    img_tensor = image_transforms(img).unsqueeze(0).to(device)  # Transform image
                    with torch.no_grad():
                        img_features = cnn_model(img_tensor).squeeze(0)  # Extract CNN features
                    batch_img_features.append(img_features)
                batch_features.append(torch.stack(batch_img_features))
            rgb_features.append(torch.stack(batch_features))  # Collect features for the key

        # Concatenate image features along the feature dimension
        rgb_features = torch.cat(rgb_features, dim=-1).to(device)

        # Combine low-dimensional data and image features
        first_state = low_dim_data[:, 0, :]  # Shape: [batch_size, total_low_dim_features]
        first_rgb_features = rgb_features[:, 0, :]  # Shape: [batch_size, cnn_feature_dim]
        states_input = torch.cat([first_state, first_rgb_features], dim=-1)

        actions_input = data['actions']  # Shape: [batch_size, seq_len, num_actions]
        
        actions_input = actions_input.to(device).float()

        # breakpoint()
        reconstructed_x, mean, logvar, prior_mean, prior_logvar = model(actions_input, states_input)
        # Check prior_logvar for NaNs or Infs
        if torch.isnan(prior_logvar).any():
            print('prior_logvar contains NaNs')
        if torch.isinf(prior_logvar).any():
            print('prior_logvar contains Infs')
        x_flat = actions_input.view(actions_input.size(0), -1)
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
        p_z_prior = Normal(prior_mean, torch.exp(0.5 * prior_logvar)) # TODO: is there a reason why it's not prior_mean and prior_logvar?
        q_z_detached = Normal(mean_detached, torch.exp(0.5 * logvar_detached))
        skill_prior_kl_loss = kl_divergence(q_z_detached, p_z_prior).mean()
        prior_loss += skill_prior_kl_loss.item()
        
        # Total loss
        total_loss = reconstruction_loss - 0.0001 * encoder_kl_loss + skill_prior_kl_loss
        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()

        # print(f"total_loss: {total_loss}, batch_idx: {batch_idx}")
        # print(f"prior_loss: {skill_prior_kl_loss}, batch_idx: {batch_idx}")
        # print(f"encoder_kl_loss: {encoder_kl_loss}, batch_idx: {batch_idx}")
        # print(f"reconstruction_loss: {reconstruction_loss}, batch_idx: {batch_idx}")
        print(f"Batch [{batch_idx + 1}], Total Loss: {total_loss:.4f}")
        print(f"Batch [{batch_idx + 1}], Prior Loss: {skill_prior_kl_loss:.4f}")
        print(f"Batch [{batch_idx + 1}], Encoder KL Loss: {encoder_kl_loss:.4f}")
        print(f"Batch [{batch_idx + 1}], Reconstruction Loss: {reconstruction_loss:.4f}")


    avg_loss_epoch = total_loss_epoch / len(dataloader)
    avg_loss_reconstruction = reconstruction_loss_epoch / len(dataloader)
    avg_loss_regularization = regularization_loss_epoch / len(dataloader)
    avg_loss_prior = prior_loss / len(dataloader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss_epoch:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Reconstruction Loss: {avg_loss_reconstruction:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average regularization Loss: {avg_loss_regularization:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average prior Loss: {avg_loss_prior:.4f}")

        