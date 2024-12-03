import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Replay buffer (not from paper)
class ReplayBuffer:
    def __init__(self, capacity, state_dim, skill_dim):
        """
        Initializes a replay buffer to store transitions.

        Args:
        - capacity (int): Maximum number of transitions the buffer can store.
        - state_dim (int): Dimensionality of the state vector.
        - skill_dim (int): Dimensionality of the skill vector (z).
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        # Preallocate memory for faster sampling
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.skills = np.zeros((capacity, skill_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, skill, reward, next_state, done):
        """
        Adds a new transition to the replay buffer.

        Args:
        - state (np.ndarray): Current state (state_dim).
        - skill (np.ndarray): Skill embedding z (skill_dim).
        - reward (float): Reward received after taking the action.
        - next_state (np.ndarray): Next state (state_dim).
        - done (bool): Whether the episode terminated at this step.
        """
        index = self.position % self.capacity  # Circular buffer
        self.states[index] = state
        self.skills[index] = skill
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        self.position += 1

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the replay buffer.

        Args:
        - batch_size (int): Number of transitions to sample.

        Returns:
        - A tuple of PyTorch tensors: (states, skills, rewards, next_states, dones)
        """
        max_index = min(self.position, self.capacity)
        indices = random.sample(range(max_index), batch_size)

        states = torch.tensor(self.states[indices], dtype=torch.float32)
        skills = torch.tensor(self.skills[indices], dtype=torch.float32)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32).unsqueeze(-1)

        return states, skills, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
        - (int): Number of stored transitions.
        """
        return min(self.position, self.capacity)