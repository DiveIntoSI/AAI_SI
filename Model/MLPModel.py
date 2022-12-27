import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        input_dim = model_params['input_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        output_dim = model_params['output_dim']

        self.W1 = nn.Linear(input_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, output_dim)

    def forward(self, input1):
        # input.shape: (batch, seq_len , embedding)
        input1 = input1.view(input1.shape[0], -1)
        return self.W2(F.relu(self.W1(input1)))
