import torch
import torch.nn as nn
from Models.SkillPriorNet import SkillPriorNet

class SkillEmbeddingAndPrior(nn.Module):
  def __init__(self, input_size, prior_size, hidden_size=128, latent_dim=10, batch_size=16):
    super(SkillEmbeddingAndPrior, self).__init__()

    self.input_size = input_size
    self.prior_size = prior_size
    self.hidden_size = hidden_size
    self.latent_dim = latent_dim
    self.sequence_length = 50 # i think horizon is 50 right?

    # Encoder LSTM layers (LSTM, mean/var linear layers)
    self.encoder_input_layer = nn.Linear(input_size, hidden_size)
    self.encoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    self.batch_norm = nn.BatchNorm1d(num_features=hidden_size)
    self.activation = nn.LeakyReLU()

    # Latent space layers for mean and variance
    self.mean = nn.Linear(hidden_size, latent_dim)
    self.var = nn.Linear(hidden_size, latent_dim)

    # Decoder LSTM: Latent vector -> Reconstructed sequences
    self.decoder_input_layer = nn.Linear(latent_dim, hidden_size)
    self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    self.fc_output = nn.Linear(hidden_size, input_size)  # Output per step should match the input size (7 in this case)

    self.skill_prior = SkillPriorNet(prior_size)

  def encode(self, x):
    hidden_output = self.encoder_input_layer(x)
    output, (hn, cn) = self.encoder_lstm(hidden_output)
    hn_last = hn[-1]  # Last layer's hidden state, shape: [batch_size, hidden_size]
    hn_last_bn = self.batch_norm(hn_last)
    hn_last_act = self.activation(hn_last_bn)

    mean = self.mean(hn_last_act)
    logvar = self.var(hn_last_act)

    return mean, logvar

  def reparameterize(self, mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

  def decode(self, z):
      z_expanded = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
      decoder_input = self.decoder_input_layer(z_expanded)  # Shape: [batch_size, sequence_length, hidden_size]
      output, _ = self.decoder_lstm(decoder_input)
      output_reshaped = output.contiguous().view(-1, self.hidden_size)  # Shape: [batch_size * sequence_length, hidden_size]

      output_bn = self.batch_norm(output_reshaped)
      output_act = self.activation(output_bn)
      output_final = self.fc_output(output_act)  # Shape: [batch_size * sequence_length, input_size]
      reconstructed_x = output_final.view(-1, self.sequence_length, self.input_size)  # Shape: [batch_size, sequence_length, input_size]
      return reconstructed_x
      
  def forward(self, x, states):
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    reconstructed_x = self.decode(z)
    prior_mean, prior_logvar = self.skill_prior(states) # x = actions...need to pass in states instead
    return reconstructed_x, mean, logvar, prior_mean, prior_logvar

#   def forward(self, x):
#       mean, logvar = self.encode(x)
#       z = self.reparameterize(mean, logvar)
#       reconstructed_x = self.decode(z)
#       prior_mean, prior_logvar = self.skill_prior(x) # x = actions...need to pass in states instead
#       return reconstructed_x, mean, logvar, prior_mean, prior_logvar
