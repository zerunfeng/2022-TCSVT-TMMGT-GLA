# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
# Copyright 2018 Antoine Miech All Rights Reserved.
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
"""Net Vlad implementation.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""
import math

import ipdb
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
  """Net Vlad module."""

  def __init__(self, cluster_size, feature_size, add_batch_norm=True):
    super().__init__()

    self.feature_size = feature_size
    self.cluster_size = cluster_size
    init_sc = (1 / math.sqrt(feature_size))
    # The `clusters` weights are the `(w,b)` in the paper
    self.clusters = nn.Parameter(init_sc * th.randn(feature_size, cluster_size))
    # The `clusters2` weights are the visual words `c_k` in the paper
    self.clusters2 = nn.Parameter(init_sc
                                  * th.randn(1, feature_size, cluster_size))
    self.add_batch_norm = add_batch_norm
    self.batch_norm = nn.BatchNorm1d(cluster_size)
    self.out_dim = cluster_size * feature_size

  def forward(self, x, mask):
    """Aggregates feature maps into a fixed size representation.

    In the following
    notation, B = batch_size, N = num_features, K = num_clusters, D =
    feature_size.

    Args:
      x: B x N x D

    Returns:
      (th.Tensor): B x DK
    """
    self.sanity_checks(x)
    mask_count = mask.sum(-1)
    mask = mask.unsqueeze(2).expand(-1, -1, self.cluster_size)
    max_sample = x.size()[1]
    x = x.view(-1, self.feature_size)  # B x N x D -> BN x D
    if x.device != self.clusters.device:
      ipdb.set_trace()
    assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x K) -> BN x K

    if self.add_batch_norm:
      assignment = self.batch_norm(assignment)

    assignment = F.softmax(assignment, dim=1)  # BN x K -> BN x K
    assignment = assignment.view(-1, max_sample,
                                 self.cluster_size)  # -> B x N x K
    assignment = assignment.masked_fill(mask == 0, 0)
    a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
    a = a_sum * self.clusters2  # B x 1 x K * 1 x D x K -> B x D x K

    assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

    x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
    vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
    vlad = vlad.transpose(1, 2)  # -> B x D x K
    vlad = vlad - a
    vlad = vlad[:, :, :-1]

    # L2 intra norm
    vlad = F.normalize(vlad)

    # flattening + L2 norm
    vlad = vlad.reshape(-1, (self.cluster_size-1) * self.feature_size)  # -> B x DK
    vlad = F.normalize(vlad)
    return vlad, assignment  # B x DK

  def sanity_checks(self, x):
    """Catch any nans in the inputs/clusters."""
    if th.isnan(th.sum(x)):
      print("nan inputs")
      ipdb.set_trace()
    if th.isnan(self.clusters[0][0]):
      print("nan clusters")
      ipdb.set_trace()
