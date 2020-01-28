import torch
import torch.nn.functional as F
from torch import nn


class CvaModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 num_tasks,
                 n_hidden,
                 device,
                 pdropout=0
                 ):
        super(CvaModel, self).__init__()

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
        # use embedding for each task
        self.embedding = nn.Embedding(num_tasks, num_context_params)
        self.zero_embeddings()

    def forward(self, x):

        # concatenate input with context parameters
        latent = x[:, self.n_in:]
        x = x[:, :self.n_in]

        latent = self.embedding(latent.flatten().long()).float()

        x = torch.cat((x.reshape(-1, 1), latent.reshape(-1, self.num_context_params)), dim=1).float()

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
            # do dropout if needed
            if self.use_do:
                x = self.do_layers[k](x)
        y = self.fc_layers[-1](x)

        return y

    def zero_embeddings(self):
        self.embedding.weight.data.fill_(0)
