import torch
import torch.nn as nn
import torch.nn.functional as F


class SAEPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        seq_len = model_params["seq_len"] # 300
        input_dim = model_params["input_dim"] # 40
        hidden_dim = model_params["hidden_dim"] # 512
        # output_dim = model_params['output_dim'] # 250
        dense_dim = model_params['dense_dim'] # 128,250,250

        # Linear Embedding
        self.embedding = Linear_Embedding(input_dim, hidden_dim)

        # 2*attnblock
        self.attnlayer = nn.TransformerEncoderLayer(hidden_dim, nhead=1)
        self.attn = nn.TransformerEncoder(self.attnlayer, 2)
        self.attn_pool = SelfAttentionPooling(hidden_dim)
        self.ffs = nn.Sequential(
            nn.Linear(hidden_dim, dense_dim[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dense_dim[0], dense_dim[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.ff = nn.Sequential(
            nn.Linear(dense_dim[1], dense_dim[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, input1):
        out1 = self.embedding(input1)
        out2 = self.attn(out1)
        out3 = self.attn_pool(out2)
        out4 = self.ffs(out3)
        if not self.training:
            return out4
        out5 = self.ff(out4)
        return out5

class Feed_Forward_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))

class Linear_Embedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, embedding_dim)
        self.norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, seq_len, input_dim)
        #  (batch * seq_len, input_dim)
        input_flat = input1.view(-1, input1.size(-1))
        #  (batch * seq_len, embedding_dim)
        out1 = F.relu(self.W(input_flat))
        #  (batch, embedding_dim, seq_len)
        out1 = out1.view(input1.size(0), input1.size(1), -1).transpose(1, 2)
        return self.norm(out1).transpose(1, 2)


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        seq_len = batch_rep.shape[1]
        softmax = nn.functional.softmax
        att_logits = self.W(batch_rep).squeeze(-1)
        att_w = softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep
