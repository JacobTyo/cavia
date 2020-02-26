import torch
import torch.nn.functional as F
from torch import nn


class TaskPredModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_hidden,
                 device
                 ):
        super(TaskPredModel, self).__init__()

        self.device = device

        n_out = n_in

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_in, n_hidden[0]))
        for k in range(len(n_hidden) - 1):
            self.fc_layers.append(nn.Linear(n_hidden[k], n_hidden[k + 1]))
        self.fc_layers.append(nn.Linear(n_hidden[-1], n_out))

    def forward(self, x):
        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y

    def get_embedding(self, x):
        with torch.no_grad():
            for k in range(2):
                x = F.relu(self.fc_layers[k](x))
        return x


class CvaModelVanilla(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 n_hidden,
                 device,
                 pdropout=0
                 ):
        super(CvaModelVanilla, self).__init__()

        self.device = device
        self.n_in = n_in
        self.pdropout = pdropout
        self.use_do = (True if self.pdropout > 0 else False)

        # fully connected layers
        self.fc_layers = nn.ModuleList()

        # dropout layers
        self.do_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(n_in + num_context_params, n_hidden[0]))
        if self.use_do:
            self.do_layers.append(nn.Dropout(p=self.pdropout))
        for k in range(len(n_hidden) - 1):
            # linear layer
            self.fc_layers.append(nn.Linear(n_hidden[k], n_hidden[k + 1]))
            # dropout
            if self.use_do:
                self.do_layers.append(nn.Dropout(p=self.pdropout))

        self.fc_layers.append(nn.Linear(n_hidden[-1], n_out))

        self.num_context_params = num_context_params

    def forward(self, x):

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
            # do dropout if needed
            if self.use_do:
                x = self.do_layers[k](x)
        y = self.fc_layers[-1](x)

        return y
