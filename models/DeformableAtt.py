#Introduced in the paper `Deformable DETR: Deformable Transformers for End-to-End Object Detection. https://arxiv.org/pdf/2010.04159

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(3233)

class DeformableAttention(nn.Module):
    def __init__(self,
                 num_heads: int = 8, #number of heads
                 d_model: int = 256, #dimensionality of the input feature vectors
                 num_points: int = 4 #number of points to sample around each query point
                 ):
        super(DeformableAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, num_heads))
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_points = num_points

        # Learnable offsets for sampling points
        self.offsets = nn.Parameter(torch.randn(num_heads, num_points, 2))
        self.to_qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(N**0.5), int(N**0.5)
        
        # Compute Q, K, V matrices
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Reshape for spatial manipulation
        x = x.view(B, H, W, C)

        # Apply offsets to get sampling coordinates
        sampling_grid = self.create_sampling_grid(H, W, self.offsets)
        sampled_features = self.sample_features(x, sampling_grid)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / (C // self.num_heads) ** 0.5
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Aggregate sampled features based on attention weights
        out = (attn_weights @ sampled_features).transpose(1, 2).reshape(B, N, C)
        out = self.fc(out)

        return out

    def create_sampling_grid(self, H, W, offsets):
        # Create a grid of coordinates and apply offsets
        base_grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1).float()
        base_grid = base_grid.unsqueeze(0).unsqueeze(0).repeat(self.num_heads, self.num_points, 1, 1, 1)
        sampling_grid = base_grid + offsets.unsqueeze(-2).unsqueeze(-2)
        return sampling_grid

    def sample_features(self, x, sampling_grid):
        B, H, W, C = x.shape
        sampling_grid = sampling_grid.view(self.num_heads, self.num_points, H, W, 2)
        sampling_grid = sampling_grid.permute(2, 3, 0, 1, 4)  # Reorder for grid_sample
        x = x.permute(0, 3, 1, 2)  # Channel first for grid_sample

        sampled_features = F.grid_sample(x, sampling_grid, mode='bilinear', align_corners=True)
        sampled_features = sampled_features.permute(0, 2, 3, 1).view(B, H * W, -1)
        return sampled_features
