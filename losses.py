"""
Loss Functions for Res_AttnGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HingeLoss(nn.Module):
    """
    Hinge Loss for GAN training
    More stable than standard GAN loss
    """
    
    def __init__(self):
        super(HingeLoss, self).__init__()
        
    def forward(self, logits, is_real):
        """
        Args:
            logits: Discrimination logits
            is_real: Boolean tensor indicating real (True) or fake (False) samples
            
        Returns:
            loss: Hinge loss
        """
        if is_real:
            # For real samples: loss = max(0, 1 - logits)
            return F.relu(1 - logits).mean()
        else:
            # For fake samples: loss = max(0, 1 + logits)
            return F.relu(1 + logits).mean()


class WassersteinLoss(nn.Module):
    """
    Wasserstein Loss
    Addresses training instability
    """
    
    def __init__(self):
        super(WassersteinLoss, self).__init__()
        
    def forward(self, logits, is_real):
        """
        Args:
            logits: Discrimination logits
            is_real: Boolean indicating real or fake
            
        Returns:
            loss: Wasserstein loss
        """
        if is_real:
            return -logits.mean()
        else:
            return logits.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss based on pre-trained VGG19
    Ensures perceptual similarity between generated and real images
    
    Args:
        layer: Which layer to use (default: 'relu5_2')
    """
    
    def __init__(self, layer='relu5_2'):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg19 = models.vgg19(pretrained=True)
        
        # Extract features up to specified layer
        layer_mapping = {
            'relu1_1': 2,
            'relu2_1': 7,
            'relu3_1': 12,
            'relu4_1': 21,
            'relu5_1': 30,
            'relu5_2': 31
        }
        
        self.features = vgg19.features[:layer_mapping[layer] + 1]
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x, y):
        """
        Args:
            x: Generated images
            y: Real images
            
        Returns:
            loss: Perceptual loss
        """
        # Normalize to ImageNet statistics
        x_norm = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        y_norm = (y + 1) / 2
        
        x_norm = (x_norm - self.mean) / self.std
        y_norm = (y_norm - self.mean) / self.std
        
        # Extract features
        x_feat = self.features(x_norm)
        y_feat = self.features(y_norm)
        
        # L1 loss on features
        return F.l1_loss(x_feat, y_feat)


class AttentionRegularizationLoss(nn.Module):
    """
    Attention Regularization Loss
    Encourages attention maps to be sparse and focused
    """
    
    def __init__(self, lambda_attn=1.0):
        super(AttentionRegularizationLoss, self).__init__()
        self.lambda_attn = lambda_attn
        
    def forward(self, attn_maps):
        """
        Args:
            attn_maps: List of attention maps
            
        Returns:
            loss: Attention regularization loss
        """
        total_loss = 0
        
        for attn_map in attn_maps:
            # attn_map shape: (batch_size, seq_len, height, width)
            
            # L1 regularization to encourage sparsity
            loss_l1 = torch.abs(attn_map).sum() / attn_map.numel()
            
            # Entropy regularization to encourage focus
            # Flatten spatial dimensions
            attn_flat = attn_map.view(attn_map.size(0), attn_map.size(1), -1)
            
            # Compute entropy over spatial dimensions
            entropy = -(attn_flat * (torch.log(attn_flat + 1e-8))).sum(dim=2).mean()
            
            total_loss = total_loss + loss_l1 + entropy
        
        return self.lambda_attn * total_loss / len(attn_maps)


class ResAttnGANLoss(nn.Module):
    """
    Combined Loss for Res_AttnGAN
    Combines discriminator, generator, and regularization losses
    """
    
    def __init__(
        self,
        loss_type='hinge',
        lambda_perceptual=1.0,
        lambda_attn=1.0,
        use_attention_loss=True
    ):
        super(ResAttnGANLoss, self).__init__()
        
        if loss_type == 'hinge':
            self.adversarial_loss = HingeLoss()
        elif loss_type == 'wasserstein':
            self.adversarial_loss = WassersteinLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.perceptual_loss = PerceptualLoss()
        self.attn_reg_loss = AttentionRegularizationLoss(lambda_attn)
        self.lambda_perceptual = lambda_perceptual
        self.use_attention_loss = use_attention_loss
        
    def discriminator_loss(self, real_logits, fake_logits):
        """
        Discriminator loss
        
        Args:
            real_logits: Logits for real images
            fake_logits: Logits for fake images
            
        Returns:
            loss: Total discriminator loss
        """
        real_loss = self.adversarial_loss(real_logits, is_real=True)
        fake_loss = self.adversarial_loss(fake_logits, is_real=False)
        
        return real_loss + fake_loss
    
    def generator_loss(self, fake_logits, generated_images, real_images, attn_maps=None):
        """
        Generator loss
        
        Args:
            fake_logits: Logits for generated images
            generated_images: Generated images
            real_images: Real images for perceptual loss
            attn_maps: Attention maps for regularization
            
        Returns:
            loss: Total generator loss
        """
        # Adversarial loss
        adv_loss = self.adversarial_loss(fake_logits, is_real=True)
        
        # Perceptual loss
        perc_loss = self.lambda_perceptual * self.perceptual_loss(
            generated_images, real_images
        )
        
        total_loss = adv_loss + perc_loss
        
        # Attention regularization
        if self.use_attention_loss and attn_maps is not None:
            attn_loss = self.attn_reg_loss(attn_maps)
            total_loss = total_loss + attn_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Useful when model is overconfident
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Sigmoid inputs
            targets: Binary targets (0 or 1)
            
        Returns:
            loss: Focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        focal_weight = self.alpha * ((1 - p_t) ** self.gamma)
        
        return (focal_weight * bce_loss).mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for text-image alignment
    """
    
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (batch_size, feature_dim)
            text_features: (batch_size, feature_dim)
            
        Returns:
            loss: Contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Create targets (diagonal should be 1, off-diagonal should be 0)
        batch_size = image_features.size(0)
        targets = torch.arange(batch_size, device=image_features.device)
        
        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        
        return (loss_i2t + loss_t2i) / 2
