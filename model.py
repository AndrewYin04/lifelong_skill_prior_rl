# model impl:

# encoder:
'''
q(z|ai) -> NN(ai) -> mean, sd..q is a gaussian distribution
'''
import torch
import torch.nn as nn

class SkillEmbeddingAndPrior(nn.Module):
  def __init__(self, input_size, hidden_size=128, latent_dim=10, batch_size=16):
    super(SkillEmbeddingAndPrior, self).__init__()

    # TODO: how can we do sequential action input to the encoder LSTM? i think lstm handles that 
    # TODO: how can we output sequential action from the decoder LSTM? i think lstm handles that 
    # TODO: rewrite LSTMs and Sequential b/c you can't do sequential w/ LSTM since LSTM output is weird
    # TODO: add skill priors DNN and function 
    self.encoder = nn.Sequential(
      nn.LSTM(input_size=input_size,
              hidden_size=hidden_size, 
              num_layers=1,
              batch_size=True),
      nn.BatchNorm1d(num_features=hidden_size),
      nn.LeakyReLU()
    )

    self.mean = nn.Linear(hidden_size, latent_dim)
    self.var = nn.Linear(hidden_size, latent_dim)

    self.decoder = nn.Sequential(

    )

class Encoder(nn.Module):

  def __init__(self, input_size, hidden_size=128, num_layers=1, batch_size=16):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    # LSTM layer
    self.lstm = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True)
    self.batch_norm_layer = nn.BatchNorm1d(hidden_size)
    self.leaky_relu_layer = nn.LeakyReLU()


  # Output size for encoder/decoder is 10
  def forward(self, x):
    # The LSTM outputs:
    # - output: the hidden states for each timestep (seq_len, batch, hidden_size)
    # - (hn, cn): hidden state and cell state from the last timestep
    output, (hn, cn) = self.lstm(x)  # Apply LSTM first

    # Reshape to apply batch normalization over the hidden size
    # Change from (batch_size, seq_len, hidden_size) to (batch_size * seq_len, hidden_size)
    output_reshaped = output.contiguous().view(-1, self.hidden_size)
    normalized_output = self.batch_norm_layer(output_reshaped)

    # Reshape back to (batch_size, seq_len, hidden_size)
    normalized_output = normalized_output.view(self.batch_size, -1, self.hidden_size)

    # Apply LeakyReLU activation after batch normalization
    normalized_output = nn.LeakyReLU()(normalized_output)

    return normalized_output.view(self.batch_size, -1, self.hidden_size), (hn, cn)


class Decoder(nn.Module):

  def __init__(self, input_size, hidden_size=128, num_layers=1, batch_size=16):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    # LSTM layer
    self.lstm = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True)
    self.batch_norm_layer = nn.BatchNorm1d(hidden_size)

  def forward(self, x):
    # The LSTM outputs:
    # - output: the hidden states for each timestep (seq_len, batch, hidden_size)
    # - (hn, cn): hidden state and cell state from the last timestep
    output, (hn, cn) = self.lstm(x)  # Apply LSTM first

    # Reshape to apply batch normalization over the hidden size
    # Change from (batch_size, seq_len, hidden_size) to (batch_size * seq_len, hidden_size)
    output_reshaped = output.contiguous().view(-1, self.hidden_size)
    normalized_output = self.batch_norm_layer(output_reshaped)

    # Reshape back to (batch_size, seq_len, hidden_size)
    normalized_output = normalized_output.view(self.batch_size, -1, self.hidden_size)

    # Apply LeakyReLU activation after batch normalization
    normalized_output = nn.LeakyReLU()(normalized_output)

    return normalized_output.view(self.batch_size, -1, self.hidden_size), (hn, cn)


# RAdam optimizer with beta one = .9, beta two = 0.999...batch size = 16, lr = 1e-3, 
# do batch norm on libero dataset

if __name__ == "__main__":
  # Example usage
  input_size = 10  # Assuming input size of 10
  hidden_size = 128
  batch_size = 16
  seq_len = 5  # Example sequence length

  encoder = Encoder(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)
  decoder = Decoder(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)

  # Example input tensor (batch_size, seq_len, input_size)
  x = torch.randn(batch_size, seq_len, input_size)

  print(x)

  # Forward pass through the encoder and decoder
  encoder_output, (hn, cn) = encoder(x)
  decoder_output, (hn_dec, cn_dec) = decoder(encoder_output)

  print(decoder_output)
