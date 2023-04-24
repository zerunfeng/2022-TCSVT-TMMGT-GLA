# Copyright 2020 Valentin Gabeur
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc.
# team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logic for the Transformer architecture used for MMT.

Code based on @huggingface implementation of Transformers:
https://github.com/huggingface/transformers
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import math

import torch
from torch import nn

logger = logging.getLogger(__name__)


def cos_sim(a, b, eps=1e-8):
  """
  calculate cosine similarity between matrix a and b
  """
  arr_a = a.cpu().detach().numpy()
  arr_b = b.cpu().detach().numpy()
  a_n, b_n = a.norm(dim=2)[:, :, None], b.norm(dim=2)[:, :, None]
  arr_a_n = a_n.cpu().detach().numpy()
  arr_b_n = b_n.cpu().detach().numpy()
  a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
  b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
  arr_a_norm = a_norm.cpu().detach().numpy()
  arr_b_norm = b_norm.cpu().detach().numpy()
  sim_mt = torch.matmul(a_norm, b_norm.permute(0, 2, 1))
  arr_sim_mt = sim_mt.cpu().detach().numpy()

  return sim_mt

def gelu(x):
  """Implementation of the gelu activation function.

  For information: OpenAI GPT's gelu is slightly different (and gives
  slightly different results):
  0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
  torch.pow(x, 3))))
  Also see https://arxiv.org/abs/1606.08415

  Args:
    x: input

  Returns:
    gelu(x)

  """
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
  return x * torch.sigmoid(x)


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
    final_mat = torch.LongTensor(final_mat).cuda()
    embeddings = self.embeddings_table[final_mat].cuda()

    return embeddings


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

# try:
#   import apex.normalization.fused_layer_norm.FusedLayerNorm as BertLayerNorm
# except (ImportError, AttributeError) as e:
#   logger.info(
#       "Better speed can be achieved with apex installed from "
#       "https://www.github.com/nvidia/apex ."
#   )
#   BertLayerNorm = torch.nn.LayerNorm

BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings."""

  def __init__(self, config):
    super(BertEmbeddings, self).__init__()
    self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                            config.hidden_size)
    self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                              config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self,
              input_ids,
              token_type_ids=None,
              position_ids=None,
              features=None):
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    if position_ids is not None:
      position_embeddings = self.position_embeddings(position_ids)
      embeddings = position_embeddings + token_type_embeddings + features
    else:
      embeddings = token_type_embeddings + features

    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class BertSimEmbeddings(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings."""

  def __init__(self, config):
    super(BertSimEmbeddings, self).__init__()
    self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                              config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self,
              input_ids,
              token_type_ids=None,
              features=None):
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embeddings = token_type_embeddings + features

    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class BertSeqEmbeddings(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings."""

  def __init__(self, config):
    super(BertSeqEmbeddings, self).__init__()
    self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                            config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self,
              position_ids=None,
              features=None):

    position_embeddings = self.position_embeddings(position_ids)
    embeddings = position_embeddings + features

    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class BertSelfAttention(nn.Module):
  """Self-attention mechanism."""

  def __init__(self, config):
    super(BertSelfAttention, self).__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.output_attentions = False

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size
                                   / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                   self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask, head_mask=None):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention
    # scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # Apply the attention mask is (precomputed for all layers in BertModel
    # forward() function)
    attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer,
               attention_probs) if self.output_attentions else (context_layer,)
    return outputs


class BertSimSelfAttention(nn.Module):
  """Self-attention mechanism."""

  def __init__(self, config):
    super(BertSimSelfAttention, self).__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.output_attentions = False

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size
                                   / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                   self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask, sim_graph, fea_shape, head_mask=None):
    b, m, seq, dim = fea_shape
    hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
    hidden_states = hidden_states.reshape(b * seq, m, dim)
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention
    # scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    sim_graph = sim_graph.unsqueeze(1)
    attention_scores = attention_scores * sim_graph

    # Apply the attention mask is (precomputed for all layers in BertModel
    # forward() function)
    attention_mask = attention_mask.permute(0, 2, 1).contiguous()
    attention_mask = attention_mask.reshape(b * seq, m)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
    attention_mask = (1.0 - attention_mask) * -10000.0
    attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer,
               attention_probs) if self.output_attentions else (context_layer,)
    return outputs


class BertGraphSelfAttention(nn.Module):
  """Graph Self-attention mechanism."""

  def __init__(self, config):
    super(BertGraphSelfAttention, self).__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.output_attentions = False

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size
                                   / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query_sim = nn.Linear(config.hidden_size, self.all_head_size)
    self.key_sim = nn.Linear(config.hidden_size, self.all_head_size)
    self.value_sim = nn.Linear(config.hidden_size, self.all_head_size)

    self.query_seq = nn.Linear(config.hidden_size, self.all_head_size)
    self.key_seq = nn.Linear(config.hidden_size, self.all_head_size)
    self.value_seq = nn.Linear(config.hidden_size, self.all_head_size)

    self.max_relative_position = config.max_relative_position

    self.relative_position_k = RelativePosition(self.attention_head_size, max_relative_position=self.max_relative_position)
    self.relative_position_v = RelativePosition(self.attention_head_size, max_relative_position=self.max_relative_position)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                   self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask, sim_graph, fea_shape, head_mask=None):

    b, m, seq, dim = fea_shape
    hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
    hidden_states = hidden_states.reshape(b * seq, m, dim)
    mixed_query_sim = self.query_sim(hidden_states)
    mixed_key_sim = self.key_sim(hidden_states)
    mixed_value_sim = self.value_sim(hidden_states)

    query_sim_layer = self.transpose_for_scores(mixed_query_sim)
    key_sim_layer = self.transpose_for_scores(mixed_key_sim)
    value_sim_layer = self.transpose_for_scores(mixed_value_sim)

    attention_scores_sim = torch.matmul(query_sim_layer, key_sim_layer.transpose(-1, -2))
    attention_scores_sim = attention_scores_sim / math.sqrt(self.attention_head_size)
    # sim_graph = sim_graph.unsqueeze(1)
    # sim_graph = (1.0 - sim_graph) * -10000.0
    # attention_scores_sim = attention_scores_sim + sim_graph

    attention_mask_sim = attention_mask.permute(0, 2, 1).contiguous()
    attention_mask_sim = attention_mask_sim.reshape(b * seq, m)
    attention_mask_sim = attention_mask_sim.unsqueeze(1).unsqueeze(2)
    attention_mask_sim = attention_mask_sim.expand(-1, 4, m, -1).byte()
    sim_graph = sim_graph.masked_fill(attention_mask_sim == 0, 0)
    sim_graph = (1.0 - sim_graph) * -10000.0
    attention_scores_sim = attention_scores_sim + sim_graph

    attention_probs_sim = nn.Softmax(dim=-1)(attention_scores_sim)

    attention_probs_sim = self.dropout(attention_probs_sim)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs_sim = attention_probs_sim * head_mask

    context_sim_layer = torch.matmul(attention_probs_sim, value_sim_layer)

    context_sim_layer = context_sim_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_sim_layer.size()[:-2] + (self.all_head_size,)
    context_sim_layer = context_sim_layer.view(*new_context_layer_shape)

    hidden_states = context_sim_layer.reshape(b, seq, m, dim)
    hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
    hidden_states = hidden_states.reshape(b*m, seq, dim)

    mixed_query_seq = self.query_seq(hidden_states)
    mixed_key_seq = self.key_seq(hidden_states)
    mixed_value_seq = self.value_seq(hidden_states)

    query_seq_layer = self.transpose_for_scores(mixed_query_seq)
    key_seq_layer = self.transpose_for_scores(mixed_key_seq)
    value_seq_layer = self.transpose_for_scores(mixed_value_seq)

    attention_scores_seq = torch.matmul(query_seq_layer, key_seq_layer.transpose(-1, -2))
    query_seq_rela_layer = query_seq_layer.permute(2, 0, 1, 3).contiguous().\
                           reshape(seq, b * m * self.num_attention_heads, self.attention_head_size)
    rela_key = self.relative_position_k(seq, seq)
    attention_scores_rela = torch.matmul(query_seq_rela_layer, rela_key.transpose(1, 2)).transpose(0, 1)
    attention_scores_rela = attention_scores_rela.contiguous().reshape(b * m, self.num_attention_heads, seq, seq)

    attention_scores_seq = attention_scores_seq + attention_scores_rela
    attention_scores_seq = attention_scores_seq / math.sqrt(self.attention_head_size)

    attention_mask_seq = attention_mask.reshape(b * m, seq)
    attention_mask_seq = attention_mask_seq.unsqueeze(1).unsqueeze(2).float()
    attention_mask_seq = (1.0 - attention_mask_seq) * -10000.0
    attention_scores_seq = attention_scores_seq + attention_mask_seq

    attention_probs_seq = nn.Softmax(dim=-1)(attention_scores_seq)

    attention_probs_seq = self.dropout(attention_probs_seq)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs_seq = attention_probs_seq * head_mask

    context_seq_layer = torch.matmul(attention_probs_seq, value_seq_layer)
    rela_value = self.relative_position_v(seq, seq)
    attention_probs_seq_rela = attention_probs_seq.permute(2, 0, 1, 3).contiguous().\
                               reshape(seq, b * m * self.num_attention_heads, seq)
    context_seq_rela_layer = torch.matmul(attention_probs_seq_rela, rela_value)
    context_seq_rela_layer = context_seq_rela_layer.transpose(0, 1).contiguous().\
                             reshape(b * m, self.num_attention_heads, seq, self.attention_head_size)
    context_seq_layer = context_seq_rela_layer + context_seq_layer
    context_seq_layer = context_seq_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_seq_layer.size()[:-2] + (self.all_head_size,)
    context_seq_layer = context_seq_layer.view(*new_context_layer_shape)

    context_seq_layer = context_seq_layer.reshape(b, m, seq, dim)

    outputs = (context_seq_layer,
               attention_probs_seq) if self.output_attentions else (context_seq_layer,)
    return outputs


class BertSeqSelfAttention(nn.Module):
  """Self-attention mechanism."""

  def __init__(self, config):
    super(BertSeqSelfAttention, self).__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.output_attentions = False

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size
                                   / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                   self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask, fea_shape, head_mask=None):
    b, m, seq, dim = fea_shape
    hidden_states = hidden_states.view(b * m, seq, dim)
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention
    # scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores = attention_scores

    # Apply the attention mask is (precomputed for all layers in BertModel
    # forward() function)
    attention_mask = attention_mask.view(b * m, seq)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
    attention_mask = (1.0 - attention_mask) * -10000.0
    attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer,
               attention_probs) if self.output_attentions else (context_layer,)
    return outputs


class BertSelfOutput(nn.Module):
  """Self-attention output."""

  def __init__(self, config):
    super(BertSelfOutput, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.layer_norm(hidden_states + input_tensor)
    return hidden_states


class BertSimSelfOutput(nn.Module):
  """Self-attention output."""

  def __init__(self, config):
    super(BertSimSelfOutput, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor, fea_shape):
    b, m, seq, dim = fea_shape
    input_tensor = input_tensor.view(b*seq, m, dim)
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.layer_norm(hidden_states + input_tensor)
    return hidden_states


class BertGraphSelfOutput(nn.Module):
  """Self-attention output."""

  def __init__(self, config):
    super(BertGraphSelfOutput, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor, fea_shape):
    b, m, seq, dim = fea_shape
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.layer_norm(hidden_states + input_tensor)
    return hidden_states

class BertSeqSelfOutput(nn.Module):
  """Self-attention output."""

  def __init__(self, config):
    super(BertSeqSelfOutput, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor, fea_shape):
    b, m, seq, dim = fea_shape
    input_tensor = input_tensor.view(b*m, seq, dim)
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.layer_norm(hidden_states + input_tensor)
    return hidden_states


class BertAttention(nn.Module):
  """Self-attention layer."""

  def __init__(self, config):
    super(BertAttention, self).__init__()
    self.self = BertSelfAttention(config)
    self.output = BertSelfOutput(config)

  def forward(self, input_tensor, attention_mask, head_mask=None):
    self_outputs = self.self(input_tensor, attention_mask, head_mask)
    attention_output = self.output(self_outputs[0], input_tensor)
    outputs = (attention_output,
              ) + self_outputs[1:]  # add attentions if we output them
    return outputs


class BertGraphAttention(nn.Module):
  """Self-attention Graph layer."""

  def __init__(self, config):
    super(BertGraphAttention, self).__init__()
    self.self = BertGraphSelfAttention(config)
    self.output = BertGraphSelfOutput(config)

  def forward(self, input_tensor, attention_mask, sim_graph, fea_shape, head_mask=None):
    self_outputs = self.self(input_tensor, attention_mask, sim_graph, fea_shape, head_mask)
    attention_output = self.output(self_outputs[0], input_tensor, fea_shape)
    outputs = (attention_output,
              ) + self_outputs[1:]  # add attentions if we output them
    return outputs

class BertSimAttention(nn.Module):
  """Self-attention layer."""

  def __init__(self, config):
    super(BertSimAttention, self).__init__()
    self.self = BertSimSelfAttention(config)
    self.output = BertSimSelfOutput(config)

  def forward(self, input_tensor, attention_mask, sim_graph, fea_shape, head_mask=None):
    self_outputs = self.self(input_tensor, attention_mask, sim_graph, fea_shape, head_mask)
    attention_output = self.output(self_outputs[0], input_tensor, fea_shape)
    outputs = (attention_output,
              ) + self_outputs[1:]  # add attentions if we output them
    return outputs


class BertSeqAttention(nn.Module):
  """Self-attention layer."""

  def __init__(self, config):
    super(BertSeqAttention, self).__init__()
    self.self = BertSeqSelfAttention(config)
    self.output = BertSeqSelfOutput(config)

  def forward(self, input_tensor, attention_mask, fea_shape, head_mask=None):
    self_outputs = self.self(input_tensor, attention_mask, fea_shape, head_mask)
    attention_output = self.output(self_outputs[0], input_tensor, fea_shape)
    outputs = (attention_output,
              ) + self_outputs[1:]  # add attentions if we output them
    return outputs


class BertIntermediate(nn.Module):
  """Fully-connected layer, part 1."""

  def __init__(self, config):
    super(BertIntermediate, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.intermediate_act_fn = ACT2FN[config.hidden_act]
    # self.intermediate_act_fn = config.hidden_act

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


class BertOutput(nn.Module):
  """Fully-connected layer, part 2."""

  def __init__(self, config):
    super(BertOutput, self).__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.layer_norm = BertLayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.layer_norm(hidden_states + input_tensor)
    return hidden_states


class BertLayer(nn.Module):
  """Complete Bert layer."""

  def __init__(self, config):
    super(BertLayer, self).__init__()
    self.attention = BertAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states, attention_mask, head_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
    attention_output = attention_outputs[0]
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,
              ) + attention_outputs[1:]  # add attentions if we output them
    return outputs


class BertGraphLayer(nn.Module):
  """Complete Bert Graph layer."""

  def __init__(self, config):
    super(BertGraphLayer, self).__init__()
    self.attention = BertGraphAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states, attention_mask, sim_graph, fea_shape, head_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask, sim_graph, fea_shape, head_mask)
    attention_output = attention_outputs[0]
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,
              ) + attention_outputs[1:]  # add attentions if we output them
    return outputs


class BertSimLayer(nn.Module):
  """Complete Bert layer."""

  def __init__(self, config):
    super(BertSimLayer, self).__init__()
    self.attention = BertSimAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states, attention_mask, sim_graph, fea_shape, head_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask, sim_graph, fea_shape, head_mask)
    attention_output = attention_outputs[0]
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,
              ) + attention_outputs[1:]  # add attentions if we output them
    return outputs


class BertSeqLayer(nn.Module):
  """Complete Bert layer."""

  def __init__(self, config):
    super(BertSeqLayer, self).__init__()
    self.attention = BertSeqAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states, attention_mask, fea_shape, head_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask, fea_shape, head_mask)
    attention_output = attention_outputs[0]
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,
              ) + attention_outputs[1:]  # add attentions if we output them
    return outputs


class BertEncoder(nn.Module):
  """Complete Bert Model (Transformer encoder)."""

  def __init__(self, config):
    super(BertEncoder, self).__init__()
    self.output_attentions = False
    self.output_hidden_states = False
    self.layer = nn.ModuleList(
        [BertLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, hidden_states, attention_mask, head_mask=None):
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer):
      if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
      hidden_states = layer_outputs[0]

      if self.output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)

    # Add last layer
    if self.output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_hidden_states:
      outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
      outputs = outputs + (all_attentions,)
    # last-layer hidden state, (all hidden states), (all attentions)
    return outputs


class BertGraphEncoder(nn.Module):
  """Graph Transformer encoder"""

  def __init__(self, config):
    super(BertGraphEncoder, self).__init__()
    self.output_attentions = False
    self.output_hidden_states = False
    self.layer = nn.ModuleList(
        [BertGraphLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, hidden_states, attention_mask, sim_graph, fea_shape, head_mask=None):
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer):
      if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_outputs = layer_module(hidden_states, attention_mask, sim_graph, fea_shape, head_mask[i])
      hidden_states = layer_outputs[0]

      if self.output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)

    # Add last layer
    if self.output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_hidden_states:
      outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
      outputs = outputs + (all_attentions,)
    # last-layer hidden state, (all hidden states), (all attentions)
    return outputs


class BertPooler(nn.Module):
  """Extraction of a single output embedding."""

  def __init__(self, config):
    super(BertPooler, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.activation = nn.Tanh()

  def forward(self, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)
    return pooled_output


class BertModel(nn.Module):
  r"""Bert Model.

  Outputs: `Tuple` comprising various elements depending on the configuration
  (config) and inputs:
      **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size,
      sequence_length, hidden_size)``
          Sequence of hidden-states at the output of the last layer of the
          model.
      **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size,
      hidden_size)``
          Last layer hidden-state of the first token of the sequence
          (classification token)
          further processed by a Linear layer and a Tanh activation function.
          The Linear
          layer weights are trained from the next sentence prediction
          (classification)
          objective during Bert pretraining. This output is usually *not* a
          good summary
          of the semantic content of the input, you're often better with
          averaging or pooling
          the sequence of hidden-states for the whole input sequence.
      **hidden_states**: (`optional`, returned when
      ``config.output_hidden_states=True``)
          list of ``torch.FloatTensor`` (one for the output of each layer +
          the output of the embeddings)
          of shape ``(batch_size, sequence_length, hidden_size)``:
          Hidden-states of the model at the output of each layer plus the
          initial embedding outputs.
      **attentions**: (`optional`, returned when
      ``config.output_attentions=True``)
          list of ``torch.FloatTensor`` (one for each layer) of shape
          ``(batch_size, num_heads, sequence_length, sequence_length)``:
          Attentions weights after the attention softmax, used to compute the
          weighted average in the self-attention heads.
  """

  def __init__(self, config):
    super(BertModel, self).__init__()

    self.config = config

    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)

    # Weights initialization
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def forward(self,
              input_ids,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              features=None):
    if attention_mask is None:
      attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to
    # [batch_size, num_heads, from_seq_length, to_seq_length]
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(
        dtype=next(self.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    head_mask = [None] * self.config.num_hidden_layers

    embedding_output = self.embeddings(input_ids,
                                       position_ids=position_ids,
                                       token_type_ids=token_type_ids,
                                       features=features)
    encoder_outputs = self.encoder(embedding_output,
                                   extended_attention_mask,
                                   head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    temp = encoder_outputs[1:]
    outputs = (
        sequence_output,
        pooled_output,
    ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    # sequence_output, pooled_output, (hidden_states), (attentions)
    return outputs


class SimGraphEncoder(nn.Module):
  def __init__(self, config):
    super(SimGraphEncoder, self).__init__()

    self.num_head = config.num_similarity_layers
    self.epsilon = config.sim_epsilon
    self.sim_heads = nn.ModuleList(
      [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.num_head)])

  def forward(self, features, attention_mask, fea_shape):
    b, m, seq, dim = fea_shape

    sim_matrixs = []
    features = features.permute(0, 2, 1, 3).contiguous()
    for _, sim_head in enumerate(self.sim_heads):
      sim_feat = sim_head(features)
      sim_feat = sim_feat.view(b*seq, m, dim)
      sim_matrixs.append(cos_sim(sim_feat, sim_feat))
    sim_matrixs = torch.stack(sim_matrixs, dim=1)
    sim_matrixs_array = sim_matrixs.cpu().detach().numpy()
    sim_agg_matrixs = sim_matrixs
    # sim_agg_matrixs = torch.mean(sim_matrixs, dim=1, keepdim=False)
    # sim_agg_matrixs_array = sim_agg_matrixs.cpu().detach().numpy()

    sim_agg_matrixs_zeros = torch.zeros_like(sim_agg_matrixs)
    sim_agg_matrixs_ones = torch.ones_like(sim_agg_matrixs)
    sim_output = torch.where(sim_agg_matrixs > self.epsilon, sim_agg_matrixs_ones, sim_agg_matrixs_zeros)

    return sim_output


class MultiModalGraphTrans(nn.Module):

  def __init__(self, config):
    super(MultiModalGraphTrans, self).__init__()

    self.config = config

    self.simgraph = SimGraphEncoder(config)
    self.embeddings = BertSimEmbeddings(config)
    self.encoder = BertGraphEncoder(config)

    # Weights initialization
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def forward(self, input_ids, attention_mask, token_type_ids, position_ids, features):
    fea_shape = features.shape
    b, m, seq, dim = fea_shape
    embedding_output = self.embeddings(input_ids,
                                       token_type_ids=token_type_ids,
                                       features=features)
    # embedding_output = embedding_output.view(b, m*seq, dim)
    sim_graph = self.simgraph(features, attention_mask, fea_shape)

    head_mask = [None] * self.config.num_hidden_layers

    # extended_attention_mask = attention_mask.view(b, seq*m).unsqueeze(1).unsqueeze(2)

    encoder_outputs = self.encoder(embedding_output,
                                   attention_mask,
                                   sim_graph,
                                   fea_shape,
                                   head_mask=head_mask)
    mmgt_output = encoder_outputs[0]
    mmgt_output = mmgt_output.reshape(b, m, seq, dim)

    return mmgt_output
