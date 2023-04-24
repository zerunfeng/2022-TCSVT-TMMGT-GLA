import math

import ipdb
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """NetVLAD layer implementation

    Args:
        num_clusters : int
            The number of clusters
        dim : int
            Dimension of descriptors
        alpha : float
            Parameter of initialization. Larger value is harder assignment.
        normalize_input : bool
            If true, descriptor-wise L2 normalization is applied to input.
    """

    def __init__(self, num_clusters=9, dim=512, alpha=100.0,
                 normalize_input=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(th.rand(num_clusters, dim))

    def forward(self, x):  # x: (N, C, H, W), H * W对应论文中的N，表示局部特征的数目，C对应论文中的D，表示特征的维度
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters,
                                        -1)  # (N, C, H, W) -> (N, num_clusters, H, W) -> (N, num_clusters, H * W)
        soft_assign = F.softmax(soft_assign, dim=1)  # (N, num_clusters, H * W)

        x_flatten = x.view(N, C, -1)  # (N, C, H, W) -> (N, C, H * W)

        # calculate residuals to each clusters
        # residual = a - b
        # a: (N, C, H * W) -> (num_clusters, N, C, H * W) -> (N, num_clusters, C, H * W)
        # b: (num_clusters, C) -> (H * W, num_clusters, C) -> (num_clusters, C, H * W)
        # residual: (N, num_clusters, C, H * W)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # soft_assign: (N, num_clusters, H * W) -> (N, num_clusters, 1, H * W)
        # (N, num_clusters, C, H * W) * (N, num_clusters, 1, H * W)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)  # (N, num_clusters, C, H * W) -> (N, num_clusters, C)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten vald: (N, num_clusters, C) -> (N, num_clusters * C)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class NetVLAD2(nn.Module):
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

  def forward(self, x):
    """Aggregates feature maps into a fixed size representation.

    In the following
    notation, B = batch_size, N = num_features, K = num_clusters, D =
    feature_size.

    Args:
      x: B x N x D

    Returns:
      (th.Tensor): B x DK
    """
    # self.sanity_checks(x)
    max_sample = x.size()[1]
    x = x.view(-1, self.feature_size)  # B x N x D -> BN x D
    # if x.device != self.clusters.device:
    #   ipdb.set_trace()
    assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x K) -> BN x K

    if self.add_batch_norm:
      assignment = self.batch_norm(assignment)

    assignment = F.softmax(assignment, dim=1)  # BN x K -> BN x K
    assignment = assignment.view(-1, max_sample,
                                 self.cluster_size)  # -> B x N x K
    a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
    a = a_sum * self.clusters2  # B x 1 x K * 1 x D x K -> B x D x K

    assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

    x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
    vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
    vlad = vlad.transpose(1, 2)  # -> B x D x K
    vlad = vlad - a

    # L2 intra norm
    vlad = F.normalize(vlad)

    # flattening + L2 norm
    vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
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


net = NetVLAD()
net2 = NetVLAD2(9, 512)
# a = th.rand(2, 512, 7, 31)
# b = net(a)
c = th.randn(2, 10, 512)
d = th.randn(2, 18, 512)
e = net2(c)
f = net2(d)

pass