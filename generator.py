"""
Generator Module for Res_AttnGAN
Multi-stage generation with residual spatial attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_modules import ResidualSpatialAttention, CBAM


class ResidualBlock(nn.Module):
    """
    Residual Block for generator
    Classic residual connection with conv layers
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (default: same as in_channels)
    """
    
    def __init__(self, in_channels, out_channels=None):
        super(ResidualBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = None
            
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            residual = self.skip(residual)
            
        out = out + residual
        out = self.relu(out)
        return out


class ResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block (Core of Res_AttnGAN)
    Combines residual connections with spatial attention
    
    Args:
        features_dim: Dimension of image features
        context_dim: Dimension of text features
        attn_dim: Attention computation dimension
    """
    
    def __init__(self, features_dim, context_dim, attn_dim=512):
        super(ResidualAttentionBlock, self).__init__()
        
        self.residual_block = ResidualBlock(features_dim)
        self.spatial_attn = ResidualSpatialAttention(
            features_dim, context_dim, attn_dim
        )
        self.cbam = CBAM(features_dim)
        
    def forward(self, features, context, context_mask=None):
        """
        Forward pass for residual attention block
        
        Args:
            features: (batch_size, features_dim, height, width)
            context: (batch_size, seq_len, context_dim)
            context_mask: (batch_size, seq_len)
            
        Returns:
            output: Attended features with residuals
            attn_map: Attention maps
        """
        # Apply residual block
        res_features = self.residual_block(features)
        
        # Apply spatial attention
        attn_features, attn_map = self.spatial_attn(res_features, context, context_mask)
        
        # Apply channel and spatial attention (CBAM)
        output = self.cbam(attn_features)
        
        return output, attn_map


class GeneratorStage(nn.Module):
    """
    Single stage of the generator
    Upsamples and refines image features
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        context_dim: Text feature dimension
        upsample_factor: Upsampling factor (default: 2)
    """
    
    def __init__(self, in_channels, out_channels, context_dim, upsample_factor=2):
        super(GeneratorStage, self).__init__()
        
        # Upsample layer
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor,
            mode='nearest'
        )
        
        # Residual blocks for refinement
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual_block = ResidualBlock(out_channels)
        
        # Residual attention mechanism
        self.residual_attn = ResidualAttentionBlock(
            out_channels, context_dim
        )
        
    def forward(self, x, context, context_mask=None):
        """
        Args:
            x: (batch_size, in_channels, height, width)
            context: (batch_size, seq_len, context_dim)
            context_mask: (batch_size, seq_len)
            
        Returns:
            output: (batch_size, out_channels, height*upsample, width*upsample)
            attn_map: Attention maps
        """
        # Upsample
        x = self.upsample(x)
        
        # Refine features
        x = self.conv(x)
        x = self.residual_block(x)
        
        # Apply attention
        output, attn_map = self.residual_attn(x, context, context_mask)
        
        return output, attn_map


class ResAttnGANGenerator(nn.Module):
    """
    Res_AttnGAN Generator
    Multi-stage generator with residual spatial attention
    
    Args:
        context_dim: Dimension of text context (default: 256)
        img_channels: Number of image channels (default: 3 for RGB)
        hidden_dim: Hidden dimension for initial features (default: 512)
    """
    
    def __init__(self, context_dim=256, img_channels=3, hidden_dim=512):
        super(ResAttnGANGenerator, self).__init__()
        
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Initial feature generation from noise and text
        self.fc = nn.Sequential(
            nn.Linear(100 + context_dim, 4 * 4 * hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 4x4 -> 8x8
        self.stage1 = GeneratorStage(
            hidden_dim, hidden_dim, context_dim, upsample_factor=2
        )
        
        # Stage 2: 8x8 -> 16x16
        self.stage2 = GeneratorStage(
            hidden_dim, hidden_dim // 2, context_dim, upsample_factor=2
        )
        
        # Stage 3: 16x16 -> 32x32
        self.stage3 = GeneratorStage(
            hidden_dim // 2, hidden_dim // 4, context_dim, upsample_factor=2
        )
        
        # Stage 4: 32x32 -> 64x64
        self.stage4 = GeneratorStage(
            hidden_dim // 4, hidden_dim // 8, context_dim, upsample_factor=2
        )
        
        # Final image generation
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, img_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, noise, sent_embs, word_embs, context_mask=None):
        """
        Forward pass for generator
        
        Args:
            noise: (batch_size, 100) - random noise
            sent_embs: (batch_size, context_dim) - sentence embeddings
            word_embs: (batch_size, seq_len, context_dim) - word embeddings
            context_mask: (batch_size, seq_len) - padding mask
            
        Returns:
            images: List of images at different scales
            attn_maps: List of attention maps at different scales
        """
        batch_size = noise.size(0)
        
        # Concatenate noise with sentence embedding
        z = torch.cat([noise, sent_embs], dim=1)  # (batch_size, 100 + context_dim)
        
        # Generate initial features
        x = self.fc(z)  # (batch_size, hidden_dim * 16)
        x = x.view(batch_size, self.hidden_dim, 4, 4)  # (batch_size, hidden_dim, 4, 4)
        
        images = []
        attn_maps = []
        
        # Stage 1
        x, attn_map = self.stage1(x, word_embs, context_mask)
        images.append(x)
        attn_maps.append(attn_map)
        
        # Stage 2
        x, attn_map = self.stage2(x, word_embs, context_mask)
        images.append(x)
        attn_maps.append(attn_map)
        
        # Stage 3
        x, attn_map = self.stage3(x, word_embs, context_mask)
        images.append(x)
        attn_maps.append(attn_map)
        
        # Stage 4
        x, attn_map = self.stage4(x, word_embs, context_mask)
        images.append(x)
        attn_maps.append(attn_map)
        
        # Final image generation (64x64)
        final_image = self.final_conv(x)
        
        return final_image, images, attn_maps


class DenseGeneratorStage(nn.Module):
    """
    Dense Generator Stage
    Enhanced version with dense connections between layers
    """
    
    def __init__(self, in_channels, out_channels, context_dim, growth_rate=32):
        super(DenseGeneratorStage, self).__init__()
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Dense block
        self.dense_layers = nn.ModuleList([
            self._make_dense_layer(in_channels + i * growth_rate, growth_rate)
            for i in range(4)
        ])
        
        self.transition = nn.Conv2d(
            in_channels + 4 * growth_rate,
            out_channels,
            1
        )
        
        self.residual_attn = ResidualAttentionBlock(
            out_channels, context_dim
        )
        
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        )
        
    def forward(self, x, context, context_mask=None):
        x = self.upsample(x)
        
        # Dense connections
        features = [x]
        for layer in self.dense_layers:
            x_concat = torch.cat(features, dim=1)
            x_new = layer(x_concat)
            features.append(x_new)
        
        x = torch.cat(features, dim=1)
        x = self.transition(x)
        
        output, attn_map = self.residual_attn(x, context, context_mask)
        
        return output, attn_map
