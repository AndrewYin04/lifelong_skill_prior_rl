# model impl:

# encoder:
'''
q(z|ai) -> NN(ai) -> mean, sd..q is a gaussian distribution
'''
import torch
import torch.nn as nn


# basic encoder
class Encoder(nn.Module):

  def __init__(self, input_size, hidden_size=128, num_layers=1):
    super(Encoder, self).__init__()
    # LSTM layer
    self.lstm = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True)
    self.batch_norm_layer = nn.BatchNorm1d(hidden_size)

  # Output size for encoder/decoder is 10
  def forward(self, x):
    # The LSTM outputs:
    # - output: the hidden states for each timestep (seq_len, batch, hidden_size)
    # - (hn, cn): hidden state and cell state from the last timestep
    output, (hn, cn) = nn.LeakyReLU(self.lstm(x))
    # batch normalization on each output (is this to ensure the probability distribution is normalized)
    normalized_output = self.batch_norm_layer(output.contiguous().view(16, self.hidden_size, -1))

    return normalized_output, (hn, cn)

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size=128, num_layers=1):
    super(Encoder, self).__init__()
    # LSTM layer
    self.lstm = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True)
    self.batch_norm_layer = nn.BatchNorm1d(hidden_size)

  # Output size for encoder/decoder is 10
  def forward(self, x):
    # The LSTM outputs:
    # - output: the hidden states for each timestep (seq_len, batch, hidden_size)
    # - (hn, cn): hidden state and cell state from the last timestep
    output, (hn, cn) = nn.LeakyReLU(self.lstm(x))
    # batch normalization on each output (is this to ensure the probability distribution is normalized)
    normalized_output = self.batch_norm_layer(output.contiguous().view(16, self.hidden_size, -1))

    return normalized_output, (hn, cn)


# RAdam optimizer with beta one = .9, beta two = 0.999...batch size = 16, lr = 1e-3, 
# do batch norm on libero dataset

