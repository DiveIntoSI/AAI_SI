import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        input_dim = model_params['input_dim']
        hidden1_dim = model_params['hidden1_dim']
        hidden2_dim = model_params['hidden2_dim']
        output_dim = model_params['output_dim']

        self.W1 = nn.Linear(input_dim, hidden1_dim)
        self.W2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.W3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, input1):
        # input.shape: (batch, seq_len , embedding)
        input1 = input1.view(input1.shape[0], -1)
        out1 = F.relu(self.W1(input1))
        out2 = F.relu(self.W2(out1))
        return self.W3(out2)

# model_params = {
#     'model': MLPModel,
#     'model_name': "MLPModel",
#     'MLPModel_params': {
#         'input_dim': 300*40,
#         'ff_hidden_dim': 512,
#         'output_dim': 250
#     }
# }
#
# model = model_params["model"]
# tmp_model = model(**model_params[model_params["model_name"] + "_params"])
