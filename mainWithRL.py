"""
TODO
    - integrate RL here

    problems
    - don't know which observation key/modality to use (https://robosuite.ai/docs/modules/environments.html#observations)
    - don't know how to flatten the observations and the dimensions of the observation
    - 
"""

from Models.PolicyNetwork import PolicyNetwork
from Models.CriticNetwork import CriticNetwork
from Models.ReplayBuffer import ReplayBuffer
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
import numpy as np
import gym
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import os
from libero_env import make_env, CFG


from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

use_arnav_code = True
if use_arnav_code:
    env = make_env(CFG("libero_object", 0, "rgb"))
else:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    task_id = 0
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(task_bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])
    # print(init_states[0].shape);quit()
    dummy_action = [0.] * 7

"""
doesnt work b/c observation() never gets called
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.flatten_space(env.env.observation_spec())
        self.state_dim = len(self.observation_space)

    def observation(self, observation):
        # return spaces.flatten(self.env.env.observation_spec(), observation)
        state_size = 0
        state_dim = 0
        state_col_arr = []
        for entry in list(observation.values()):
            # state_size += 1
            try:
                for entry2 in entry:
                    try:
                        for entry3 in entry2:
                            state_size += 1
                            state_col_arr.append(entry3)
                    except:
                        state_size += 1
                        state_col_arr.append(entry2)
            except:
                state_size += 1
                state_col_arr.append(entry)
            # state_dim = state_size
            # print(state_size)
            # state_size = 0
        state_dim = state_size
        raise ''
        print('START',np.array(state_col_arr),'END');quit()
        return np.array(state_col_arr)
"""
# env.observation_space OrderedDict space
# from gym.wrappers import FlattenObservation
# env = FlattenObservation(env)
# Box
for step in range(10):
    # obs OrderedDict
    obs, reward, done, info = env.step(dummy_action)
    # obs vector
    # print(len(list(obs.values())));quit()

breakpoint()
state_dim = env.observation_space.shape[0]
action_dim = list(env.env.action_spec)[0].shape[0]



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# env configuration for RL
# env = gym.make('FrozenLake-v1')
# breakpoint()
# print(env.env.observation_spec())
# state_dim = env.env.observation_spec().shape[0]
# print(env.sim.get_state().flatten().shape)
# print(vars(env.sim))
print(env.env.observation_spec())
#uit()
state_size = 0
state_dim = 0
for entry in list(env.env.observation_spec().values()):
    # state_size += 1
    try:
        for entry2 in entry:
            try:
                for entry3 in entry2:
                    state_size += 1
            except:
                state_size += 1
    except:
        state_size += 1
    # state_dim = state_size
    # print(state_size)
    # state_size = 0

state_dim = state_size
# state_dim = env.env.observation_space.shape[0]
action_dim = list(env.env.action_spec)[0].shape[0]
print(state_dim)
print(action_dim)

# Using VAE model defined above...
dataset_path = 'libero/libero/bddl_files/libero_spatial/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5'
obs_modality = {
    'low_dim': ['ee_ori', 'ee_pos', 'ee_states', 'gripper_states', 'joint_states'],
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
prior_dim = 21  # Adjusted since we no longer have import libero_env # is this how to import a file?image features
model = SkillEmbeddingAndPrior(input_dim, prior_dim).to(device)
print("State dim: " + str(state_dim)) # should be 21 right?
print("Action dim: " + str(action_dim)) # should be 7 right?

reconstruction_loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

prior_loss_array = []
total_loss_array = []

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

        actions_input = data['actions']  # Shape: [batch_size, seq_len, num_actions]
        actions_input = actions_input.to(device).float()

        # Pass data through the model
        reconstructed_x, mean, logvar, prior_mean, prior_logvar = model(actions_input, states_input)

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

# RL STUFF

# do this reparameterization between policy and critic model runs in RL loop (ASK DR.A why need tanh?)
# this is basically "get_action()"
def reparameterizeForRL(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std
    z = torch.tanh(z) * 2
    return z

# TODO: Now with these RL models, gotta make the RL loop (via the complex math eqs in paper...)
# We need one policy model, 2 critics (take min of the 2 Q-vals for each iter), and 2 target critics (again take min)
# Must use KL divergence loss and other complex math eq's to do the RL steps

# Hyperparameters
γ = 0.99       # Discount factor
τ = 0.005      # Target network update rate
δ = 1.0        # Target divergence
λ_π = 3e-4     # Policy learning rate
λ_Q = 3e-4     # Critic learning rate
λ_α = 3e-4     # Alpha learning rate
H = 50         # Skill duration (Ask if 50 is fine?)
batch_size = 256

# Initialize models
policy_net = PolicyNetwork(state_dim)
critic_net = CriticNetwork(state_dim)
target_critic_net = CriticNetwork(state_dim)
target_critic_net.load_state_dict(critic_net.state_dict())

# turning off gradients in model since it has already been pretrained
for param in model.parameters():
    param.requires_grad = False

# Optimizers
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=λ_π)
critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=λ_Q)

# Automatic entropy tuning
log_alpha = torch.tensor(np.log(1.0), requires_grad=True)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=λ_α)
gradient_steps_per_iteration = 1 # change as necessary

def execute_skill(z_t, state, env, H, model):
    skill_reward = 0
    next_state = state
    done = False

    for _ in range(H):
        action = model.decode(z_t)  # Pre-trained skill decoder
        next_state, reward, done, info = env.step(action)
        skill_reward += reward
        if done:
            break

    return next_state, skill_reward, done

replay_buffer = ReplayBuffer()
num_iterations = 1000  # Define as neededimport libero_env # is this how to import a file?

for iteration in range(num_iterations):
    state = env.reset()
    done = False

    # while not done:
    # Sample skill z_t from policy
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    # ask DR.A: if i need to run policy_net and buffer.push() H times too? or just the decoder in execute_skill?
    with torch.no_grad():
        mu, logvar = policy_net(state_tensor)
        # dist = Normal(mu, std)
        # z_t = dist.sample()
        # z_t_np = z_t.squeeze(0).numpy()
        z_t_np = reparameterizeForRL(mu, logvar)

    # Execute skill and accumulate rewards
    next_state, skill_reward, done = execute_skill(z_t_np, state, env, H)

    # Store transition
    replay_buffer.push(state, z_t_np, skill_reward, next_state)

    state = next_state

    if len(replay_buffer) > batch_size:
        # Perform gradient updates
        for _ in range(gradient_steps_per_iteration):
            # Sample batch
            state_batch, z_batch, reward_batch, next_state_batch = replay_buffer.sample(batch_size)
            state_batch = torch.FloatTensor(state_batch)
            z_batch = torch.FloatTensor(z_batch)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(next_state_batch)

            # Critic update
            with torch.no_grad():
                mu_next, log_var_next = policy_net(next_state_batch)
                # z_next = Normal(mu_next, std_next).rsample()
                z_next = reparameterizeForRL(mu_next, log_var_next)
                Q_target = target_critic_net(next_state_batch, z_next)

                mu_prior_next, logvar_prior_next = model.skill_prior_net.forward(next_state_batch) # ASK DR.A if this syntax is correct?
                D_KL_next = kl_divergence(Normal(mu_next, torch.exp(0.5 * log_var_next)), Normal(mu_prior_next, torch.exp(0.5 * logvar_prior_next))).sum(dim=1, keepdim=True)

                Q_target = reward_batch.unsqueeze(1) + γ * (Q_target - torch.exp(log_alpha) * D_KL_next)

            Q_value = critic_net(state_batch, z_batch)
            critic_loss = 0.5 * (Q_value - Q_target).pow(2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Policy update (ask dr.A why use state_Batch for policy but use next_state_batch for critic...)
            mu, logvar = policy_net(state_batch)
            dist = Normal(mu, torch.exp(0.5 * logvar))
            # z_tilde = dist.rsample()
            z_tilde = reparameterizeForRL(mu, logvar)
            Q_value = critic_net(state_batch, z_tilde)

            mu_prior, logvar_prior = model.skill_prior_net.forward(state_batch)
            D_KL = kl_divergence(dist, Normal(mu_prior, torch.exp(0.5 * logvar_prior))).sum(dim=1, keepdim=True)

            policy_loss = (-Q_value + torch.exp(log_alpha) * D_KL).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # Alpha update
            alpha_loss = (torch.exp(log_alpha) * (D_KL.detach() - δ)).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # Target network update
            for target_param, param in zip(target_critic_net.parameters(), critic_net.parameters()):
                target_param.data.copy_(τ * param.data + (1 - τ) * target_param.data)
