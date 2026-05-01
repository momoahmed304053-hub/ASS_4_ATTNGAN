"""
Discriminator Module for Res_AttnGAN
Multi-scale discriminators with spectral normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(nn.Module):
    """
    Spectral Normalization for Discriminator
    Improves training stability
    """
    
    def __init__(self, module, n_power_iterations=1, eps=1e-12):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        if isinstance(self.module, (nn.Linear, nn.Conv2d)):
            # Initialize u randomly
            if isinstance(self.module, nn.Conv2d):
                num_input = self.module.weight.size(1)
                for _ in self.module.weight.shape[1:]:
                    num_input = num_input * _ 
            else:
                num_input = self.module.weight.size(1)
            
            u = torch.randn(1, num_input)
            self.register_buffer('u', u)
            
    def forward(self, x):
        return self.module(x)
    
    def _normalize_weights(self):
        """Apply spectral normalization"""
        if isinstance(self.module, (nn.Linear, nn.Conv2d)):
            w = self.module.weight
            
            # Reshape weight matrix
            if isinstance(self.module, nn.Conv2d):
                w_shape = w.shape
                w = w.view(w_shape[0], -1)
            
            # Power iteration
            u = self.u
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(w.t(), u.t()), dim=0)
                u = F.normalize(torch.mv(w, v), dim=0)
            
            # Compute spectral norm
            sigma = torch.dot(u.view(-1), torch.mv(w, v))
            
            # Normalize weight
            self.module.weight.data = w / (sigma + self.eps)
            
            if isinstance(self.module, nn.Conv2d):
                self.module.weight.data = self.module.weight.data.view(w_shape)
            
            # Update u
            self.u.copy_(u)


class SpectralNormConv2d(nn.Conv2d):
    """
    Conv2d with Spectral Normalization
    """
    
    def __init__(self, *args, **kwargs):
        super(SpectralNormConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('u', torch.randn(1, self.out_channels))
        
    def forward(self, x):
        # Apply spectral norm to weight
        self._normalize_weight()
        return super(SpectralNormConv2d, self).forward(x)
    
    def _normalize_weight(self):
        w = self.weight
        w_shape = w.shape
        w = w.view(w_shape[0], -1)
        
        u = self.u
        v = F.normalize(torch.mv(w.t(), u.t()), dim=0)
        u = F.normalize(torch.mv(w, v), dim=0)
        
        sigma = torch.dot(u.view(-1), torch.mv(w, v))
        self.weight.data = w.view(w_shape) / (sigma + 1e-12)
        self.u.copy_(u)


class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block
    Downsampling block with residual connections
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        downsample: Whether to downsample (default: True)
    """
    
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscriminatorBlock, self).__init__()
        
        stride = 2 if downsample else 1
        
        self.conv1 = SpectralNormConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = SpectralNormConv2d(
            out_channels, out_channels, 3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if downsample or in_channels != out_channels:
            self.skip = SpectralNormConv2d(
                in_channels, out_channels, 1, stride=stride
            )
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


class ImageDiscriminator(nn.Module):
    """
    Image Discriminator for single scale
    Discriminates between real and generated images
    
    Args:
        img_channels: Number of image channels (3 for RGB)
        conv_dim: Base number of convolution filters (default: 64)
    """
    
    def __init__(self, img_channels=3, conv_dim=64):
        super(ImageDiscriminator, self).__init__()
        
        self.layer1 = SpectralNormConv2d(img_channels, conv_dim, 4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer2 = DiscriminatorBlock(conv_dim, conv_dim * 2, downsample=True)
        self.layer3 = DiscriminatorBlock(conv_dim * 2, conv_dim * 4, downsample=True)
        self.layer4 = DiscriminatorBlock(conv_dim * 4, conv_dim * 8, downsample=True)
        
        # Classification head
        self.conv_final = SpectralNormConv2d(conv_dim * 8, 1, 4)
        
    def forward(self, images):
        """
        Args:
            images: (batch_size, img_channels, height, width)
            
        Returns:
            logits: (batch_size, 1, 1, 1) - discrimination logits
            features: List of intermediate features
        """
        features = []
        
        x = self.layer1(images)
        x = self.relu(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        logits = self.conv_final(x)
        
        return logits, features


class TextContextDiscriminator(nn.Module):
    """
    Text-Image Context Discriminator
    Ensures text-image correspondence
    
    Args:
        img_channels: Number of image channels
        context_dim: Dimension of text context
    """
    
    def __init__(self, img_channels=3, context_dim=256):
        super(TextContextDiscriminator, self).__init__()
        
        self.img_discriminator = ImageDiscriminator(img_channels)
        
        # Context processing
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        
        # Joint classification
        self.joint_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
    def forward(self, images, context):
        """
        Args:
            images: (batch_size, img_channels, height, width)
            context: (batch_size, context_dim)
            
        Returns:
            logits: Joint text-image discrimination logits
        """
        # Image features
        _, img_features = self.img_discriminator(images)
        
        # Use last feature map
        img_feat = img_features[-1]  # (batch_size, conv_dim*8, h, w)
        img_feat = F.adaptive_avg_pool2d(img_feat, 1).view(img_feat.size(0), -1)
        
        # Context features
        context_feat = self.context_mlp(context)
        
        # Combine features
        combined = img_feat + context_feat
        logits = self.joint_classifier(combined)
        
        return logits


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator
    Discriminates at multiple scales for better gradient flow
    
    Args:
        num_scales: Number of scales to discriminate (default: 4)
        img_channels: Number of image channels
        context_dim: Dimension of text context
    """
    
    def __init__(self, num_scales=4, img_channels=3, context_dim=256):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        self.img_discriminators = nn.ModuleList([
            ImageDiscriminator(img_channels) for _ in range(num_scales)
        ])
        
        self.text_discriminators = nn.ModuleList([
            TextContextDiscriminator(img_channels, context_dim) 
            for _ in range(num_scales)
        ])
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, images_list, context):
        """
        Args:
            images_list: List of images at different scales or single image
            context: (batch_size, context_dim)
            
        Returns:
            img_logits: List of image discrimination logits
            text_logits: List of text-image discrimination logits
            features_list: List of features from each scale
        """
        if not isinstance(images_list, list):
            images_list = [images_list]
            
        # If only one image provided, create multi-scale versions
        if len(images_list) == 1:
            img = images_list[0]
            images_list = [img]
            for _ in range(self.num_scales - 1):
                img = self.downsample(img)
                images_list.append(img)
        
        img_logits = []
        text_logits = []
        features_list = []
        
        for i in range(self.num_scales):
            # Image discrimination
            logits, features = self.img_discriminators[i](images_list[i])
            img_logits.append(logits)
            features_list.append(features)
            
            # Text-image discrimination
            text_logits_i = self.text_discriminators[i](images_list[i], context)
            text_logits.append(text_logits_i)
        
        return img_logits, text_logits, features_list
