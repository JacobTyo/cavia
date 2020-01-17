import torch
import torch.nn.functional as F
from torch import nn


class GameTaskModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden,
                 device
                 ):
        super(GameTaskModel, self).__init__()

        self.device = device
        self.n_in = n_in

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, n_hidden[0]))
        self.layers.append(nn.RNN(n_hidden[0], n_hidden[1]))
        self.layers.append(nn.Linear(n_hidden[1], n_out))

        self.last_hidden = torch.zeros(1, 1, n_hidden[1]).to(device)

    def forward(self, x):
        x = F.relu(self.layers[0](x))
        x, h = self.layers[1](x, self.last_hidden)
        self.last_hidden = h.detach()
        y = F.softmax(self.layers[2](x))

        return y
