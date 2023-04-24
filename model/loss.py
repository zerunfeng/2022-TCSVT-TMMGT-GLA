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
"""Training losses.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(p, q):
	return (p * th.log(p/q)).sum(dim=1)


class MaxMarginRankingLoss(nn.Module):
  """Implementation of the Max-margin ranking loss."""

  def __init__(self, margin=1, fix_norm=True):
    super().__init__()
    self.fix_norm = fix_norm
    self.loss = th.nn.MarginRankingLoss(margin)
    self.margin = margin
    self.log_sigmoid = nn.LogSigmoid()

  def forward(self, x, text_assign, vid_assign, text_mask, vid_mask):
    n = x.size()[0]

    x1 = th.diag(x)
    x1 = x1.unsqueeze(1)
    x1 = x1.expand(n, n)
    x1 = x1.contiguous().view(-1, 1)
    x1 = th.cat((x1, x1), 0)

    x2 = x.view(-1, 1)
    x3 = x.transpose(0, 1).contiguous().view(-1, 1)

    x2 = th.cat((x2, x3), 0)
    max_margin = F.relu(self.margin - (x1 - x2))

    if self.fix_norm:
      # remove the elements from the diagonal
      keep = th.ones(x.shape) - th.eye(x.shape[0])
      keep1 = keep.view(-1, 1)
      keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
      keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
      if x1.is_cuda:
        keep_idx = keep_idx.cuda()
      x1_ = th.index_select(x1, dim=0, index=keep_idx)
      x2_ = th.index_select(x2, dim=0, index=keep_idx)
      max_margin = F.relu(self.margin - (x1_ - x2_))

    text_cluster_loss = (text_assign * text_assign).sum(dim=1)
    text_cluster_loss = self.log_sigmoid(text_cluster_loss)
    text_cluster_loss = text_cluster_loss.masked_fill(text_mask == 0, 0)
    text_cluster_loss = -text_cluster_loss.sum(dim=1) / text_mask.sum(dim=1).float()

    vid_cluster_loss = (vid_assign * vid_assign).sum(dim=1)
    vid_cluster_loss = self.log_sigmoid(vid_cluster_loss)
    vid_cluster_loss = vid_cluster_loss.masked_fill(vid_mask == 0, 0)
    vid_cluster_loss = -vid_cluster_loss.sum(dim=1) / vid_mask.sum(dim=1).float()

    text_distri = th.div(text_assign.sum(dim=2), text_mask.sum(dim=1).float().unsqueeze(1))
    text_uni_distri = th.div(th.ones_like(text_distri).float(), text_mask.sum(dim=1).float().unsqueeze(1))
    text_kl_loss = kl_divergence(text_distri.softmax(dim=-1), text_uni_distri.softmax(dim=-1))
    # text_kl_loss2 = F.kl_div(text_uni_distri.softmax(dim=-1).log(), text_distri.softmax(dim=-1), reduction='none').sum(dim=-1)


    vid_distri = th.div(vid_assign.sum(dim=2), vid_mask.sum(dim=1).float().unsqueeze(1))
    vid_uni_distri = th.div(th.ones_like(vid_distri).float(), vid_mask.sum(dim=1).float().unsqueeze(1))
    vid_kl_loss = kl_divergence(vid_distri.softmax(dim=-1), vid_uni_distri.softmax(dim=-1))

    cluster_loss = text_cluster_loss + vid_cluster_loss + text_kl_loss + vid_kl_loss
    cluster_loss = cluster_loss.mean()

    triplet_loss = max_margin.mean()

    # return triplet_loss + 0.001*cluster_loss
    return triplet_loss


class InfoNceLoss(nn.Module):
  """Implementation of the noise-constrastive estimation loss."""

  def __init__(self):
    super().__init__()
    self.loss = th.nn.CrossEntropyLoss(reduction='mean')

  def forward(self, x):
    n = x.size()[0]
    target = th.arange(n)
    if x.is_cuda:
      target = target.cuda()

    return self.loss(x, target) + self.loss(th.transpose(x, 0, 1), target)
