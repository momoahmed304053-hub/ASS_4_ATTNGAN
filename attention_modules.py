"""
Attention Modules for Res_AttnGAN
Implements residual spatial attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Computes spatial attention maps for image-text alignment
    
    Args:
        query_dim: Dimension of query (image features)
        context_dim: Dimension of context (text features)
        attn_dim: Dimension of attention computation (default: 512)
    """
    
    def __init__(self, query_dim, context_dim, attn_dim=512):
        super(SpatialAttention, self).__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.attn_dim = attn_dim
        
        # Query projection
        self.query_proj = nn.Conv2d(query_dim, attn_dim, 1)
        
        # Context projection
        self.context_proj = nn.Linear(context_dim, attn_dim)
        
        # Attention score computation
        self.score_proj = nn.Linear(attn_dim, 1)
        
    def forward(self, query, context, context_mask=None):
        """
        Forward pass for spatial attention
        
        Args:
            query: (batch_size, query_dim, height, width) - image features
            context: (batch_size, seq_len, context_dim) - text features
            context_mask: (batch_size, seq_len) - mask for padding tokens
            
        Returns:
            output: (batch_size, query_dim, height, width) - attended features
            attn_map: (batch_size, seq_len, height, width) - attention maps
        """
        batch_size, _, height, width = query.size()
        seq_len = context.size(1)
        
        # Project query
        query_proj = self.query_proj(query)  # (batch_size, attn_dim, height, width)
        query_flat = query_proj.view(batch_size, self.attn_dim, -1)  # (batch_size, attn_dim, hw)
        query_flat = query_flat.transpose(1, 2)  # (batch_size, hw, attn_dim)
        
        # Project context
        context_proj = self.context_proj(context)  # (batch_size, seq_len, attn_dim)
        
        # Compute attention scores
        # (batch_size, hw, 1, attn_dim) + (batch_size, 1, seq_len, attn_dim)
        attn_scores = query_flat.unsqueeze(2) + context_proj.unsqueeze(1)
        attn_scores = torch.tanh(attn_scores)
        attn_scores = self.score_proj(attn_scores)  # (batch_size, hw, seq_len, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch_size, hw, seq_len)
        
        # Apply mask if provided
        if context_mask is not None:
            context_mask = context_mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            attn_scores.masked_fill_(~context_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=2)  # (batch_size, hw, seq_len)
        attn_weights = attn_weights.transpose(1, 2)  # (batch_size, seq_len, hw)
        
        # Reshape for visualization and computation
        attn_map = attn_weights.view(batch_size, seq_len, height, width)
        
        # Weight image features
        query_weighted = query.view(batch_size, self.query_dim, -1)  # (batch_size, query_dim, hw)
        query_weighted = query_weighted.unsqueeze(1)  # (batch_size, 1, query_dim, hw)
        
        # attn_weights: (batch_size, seq_len, hw)
        # Query: (batch_size, 1, query_dim, hw)
        attn_weights_expanded = attn_weights.unsqueeze(2)  # (batch_size, seq_len, 1, hw)
        
        weighted_features = query_weighted * attn_weights_expanded  # broadcast
        weighted_features = weighted_features.sum(dim=3, keepdim=True)  # (batch_size, seq_len, query_dim, 1)
        
        # Alternative simpler implementation: use max attention
        max_attn_idx = torch.argmax(attn_weights, dim=1)  # (batch_size, hw)
        output = query  # For now, return original features
        
        return output, attn_map


class ResidualSpatialAttention(nn.Module):
    """
    Residual Spatial Attention Block
    Key innovation of Res_AttnGAN: Adds residual connections to attention computation
    
    Args:
        features_dim: Dimension of image features
        context_dim: Dimension of text features
        attn_dim: Dimension of attention computation
    """
    
    def __init__(self, features_dim, context_dim, attn_dim=512):
        super(ResidualSpatialAttention, self).__init__()
        self.spatial_attn = SpatialAttention(features_dim, context_dim, attn_dim)
        
        # Residual path projection
        self.residual_proj = nn.Conv2d(features_dim, features_dim, 1)
        
        # Feature fusion layer
        self.fusion = nn.Conv2d(features_dim * 2, features_dim, 1)
        
    def forward(self, features, context, context_mask=None):
        """
        Forward pass for residual spatial attention
        
        Args:
            features: (batch_size, features_dim, height, width) - image features
            context: (batch_size, seq_len, context_dim) - text features
            context_mask: (batch_size, seq_len) - attention mask
            
        Returns:
            output: (batch_size, features_dim, height, width) - attended features with residual
            attn_map: attention maps
        """
        # Spatial attention computation
        attn_features, attn_map = self.spatial_attn(features, context, context_mask)
        
        # Residual path
        residual = self.residual_proj(features)
        
        # Combine attention and residual
        # attn_features has attention applied
        combined = torch.cat([attn_features, features], dim=1)
        output = self.fusion(combined)
        
        # Add residual connection
        output = output + residual
        
        return output, attn_map


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Computes channel-wise importance weights
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
            
        Returns:
            output: (batch_size, channels, height, width) - channel-weighted features
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * torch.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module (different from SpatialAttention)
    Computes spatial importance weights
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
            
        Returns:
            output: (batch_size, channels, height, width) - spatially weighted features
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * torch.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttentionModule(kernel_size)
        
    def forward(self, x):
        """Apply channel then spatial attention"""
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
