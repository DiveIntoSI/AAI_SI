import torch
import torch.nn as nn
import torch.nn.functional as F


class STModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        input_dim = model_params["input_dim"]
        seq_len = model_params["seq_len"]
        dropout_p = model_params['dropout_p']

        # Linear Embedding
        self.embedding1 = Linear_Embedding(input_dim, input_dim)  # 线性层还是conv1d？

        # Time attention layer
        time_head_num = model_params["time_head_num"]  # 8 * 25 = 200
        self.time_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                               num_heads=time_head_num)
        self.add_and_norm1 = Add_And_Normalization_Module(input_dim)

        # dropout
        self.dropout1 = nn.Dropout(dropout_p)

        # Linear Embedding
        self.embedding2 = nn.Linear(seq_len, seq_len)  # 线性层还是conv1d？

        # Spatial attention layer
        spatial_head_num = model_params["spatial_head_num"]  # 10 * 9 = 90
        self.spatial_attn = nn.MultiheadAttention(embed_dim=seq_len,
                                                  num_heads=spatial_head_num)
        self.add_and_norm2 = Add_And_Normalization_Module(seq_len)

        # dropout
        self.dropout2 = nn.Dropout(dropout_p)

        # fc layer
        fc_hidden_dim = model_params["fc_hidden_dim"]
        self.fc = Feed_Forward_Module(input_dim * seq_len, fc_hidden_dim)

    def forward(self, data):
        # (32, 90, 200)
        # linear embedding
        embedded1 = self.embedding1(data)

        # Time attention layer
        # (90, 32, 200)
        embedded1_trans = embedded1.transpose(0, 1)
        attn1, attn_weight1 = self.time_attn(embedded1_trans, embedded1_trans, embedded1_trans)
        # (32, 90, 200)
        attn1_trans = attn1.transpose(0, 1)
        out1 = self.add_and_norm1(embedded1, attn1_trans)

        # drop
        # out1 = self.dropout1(out1)

        # linear embedding
        # (32, 200, 90)
        out1 = out1.transpose(1, 2)
        embedded2 = self.embedding2(out1)

        # Spatial attention layer
        # (200, 32, 90)
        embedded2_trans = embedded2.transpose(0, 1)
        attn2, attn_weight2 = self.spatial_attn(embedded2_trans, embedded2_trans, embedded2_trans)
        # (32, 200, 90)
        attn2_trans = attn2.transpose(0, 1)
        out2 = self.add_and_norm2(embedded2, attn2_trans)

        # drop
        # out2 = self.dropout2(out2)

        # fc linear
        # (32, 90*200)
        out2 = out2.transpose(1, 2).contiguous().view(out2.size(0), -1)
        out3 = self.fc(out2)
        prob = torch.sigmoid(out3)
        return prob


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (32, 90, 200)
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super().__init__()

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, 1)

    def forward(self, input1):
        # input.shape: (batch, seq_len, embedding)
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
