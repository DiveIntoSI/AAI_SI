import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        output_hidden_dim = model_params['output_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, output_hidden_dim)

    def forward(self, input1):
        # input.shape: (batch, seq_len * embedding)
        return self.W2(F.relu(self.W1(input1)))
