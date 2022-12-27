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
        out1 = self.W1(input1)
        out1 = F.relu(out1)
        return self.W2(out1)



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