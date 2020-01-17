import torch
import torch.nn.functional as F
from torch import nn


class GameOptModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden,
                 device
                 ):
        super(GameOptModel, self).__init__()

        self.device = device
        self.n_in = n_in

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_in, n_hidden[0]))
        for k in range(len(n_hidden) - 1):
            self.fc_layers.append(nn.Linear(n_hidden[k], n_hidden[k + 1]))
        self.fc_layers.append(nn.Linear(n_hidden[-1], n_out))

    def forward(self, x):
        # concatenate input with context parameters
        # latent = x[:, self.n_in:]
        # x = x[:, :self.n_in]
        #
        # x = torch.cat((x.reshape(-1, 1), latent.reshape(-1, 2)), dim=1).float()

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y
