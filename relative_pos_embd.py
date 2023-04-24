import math

import torch
from torch import nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class RelativePosition(nn.Module):
    def __init__(self, hidden_size, max_relative_position):
        super().__init__()
        self.num_units = hidden_size
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.randn(max_relative_position * 2 + 1, hidden_size))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, batch_size=10):
        "Take in model size and number of heads."
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.batch_size = batch_size

        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        # self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.relative_position_k = RelativePosition(self.head_dim, max_relative_position=16)
        self.relative_position_v = RelativePosition(self.head_dim, max_relative_position=16)

        self.scale = math.sqrt(self.batch_size)

    def forward(self, query, key, value):
        # embedding
        # query, key, value = [batch_size, len, hid_dim]
        query, key, value = [l(x).view(self.batch_size, -1, self.d_model) for l, x in
                             zip(self.linears, (query, key, value))]

        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        # Self-Attention
        # r_q1, r_k1 = [batch_size, len, n_heads, head_dim]
        r_q1 = query.view(self.batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(self.batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, self.batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(self.batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        attn = self.dropout(torch.softmax(attn, dim=-1))
        # attn = [batch_size, n_heads, len, len]
        r_v1 = value.view(self.batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, self.batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(self.batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(self.batch_size * len_q, self.d_model)
        # x = [batch size * query len, hid dim]

        return self.linears[-1](x)


# range_vec_q = torch.arange(10)
# range_vec_k = torch.arange(10)
#
# distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#
# distance_mat_clipped = torch.clamp(distance_mat, -4, 4)
# final_mat = distance_mat_clipped + 4
relative_position_k = RelativePosition(512, max_relative_position=16)
net = RelativeMultiHeadAttention(512, 4)
x = torch.randn(10, 30, 512)
y = net(x, x, x)


a=1