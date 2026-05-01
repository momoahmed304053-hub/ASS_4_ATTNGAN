"""
Training utilities and training loop for Res_AttnGAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from pathlib import Path
import json
from datetime import datetime


class TrainingConfig:
    """Configuration for training"""
    
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 100
        self.num_workers = 4
        self.img_size = 64
        
        # Optimizers
        self.lr_generator = 0.0002
        self.lr_discriminator = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        
        # Loss weights
        self.lambda_perceptual = 1.0
        self.lambda_attn = 1.0
        self.lambda_text = 1.0
        
        # Training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_interval = 10
        self.log_interval = 100
        
        # Paths
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'
        
    def to_dict(self):
        return self.__dict__


class Trainer:
    """
    Trainer for Res_AttnGAN
    Handles training loop, validation, and checkpoint saving
    """
    
    def __init__(
        self,
        generator,
        discriminators,
        text_encoder,
        loss_fn,
        device='cuda',
        config=None
    ):
        self.generator = generator.to(device)
        self.discriminators = discriminators
        self.text_encoder = text_encoder
        self.loss_fn = loss_fn
        self.device = device
        
        self.config = config or TrainingConfig()
        
        # Setup optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr_generator,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        # Collect discriminator parameters
        discriminator_params = []
        for disc in self.discriminators:
            discriminator_params.extend(disc.parameters())
        
        self.optimizer_d = optim.Adam(
            discriminator_params,
            lr=self.config.lr_discriminator,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        # Create directories
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.config.log_dir).mkdir(exist_ok=True)
        
        # Training metrics
        self.metrics = {
            'loss_d': [],
            'loss_g': [],
            'loss_perceptual': [],
            'loss_attn': []
        }
        
    def train_step(self, batch):
        """
        Single training step
        
        Args:
            batch: Data batch dict with 'image' and 'caption'
            
        Returns:
            metrics: Dict of loss metrics
        """
        # Handle both 'image' and 'images' keys
        if 'image' in batch:
            real_images = batch['image'].to(self.device)
        else:
            real_images = batch['images'].to(self.device)
        
        # Encode text
        if 'caption' in batch:
            captions = batch['caption']
        else:
            captions = batch['captions']
        
        # For now, use dummy embeddings
        batch_size = real_images.size(0)
        sent_embs = torch.randn(batch_size, 256).to(self.device)
        word_embs = torch.randn(batch_size, 20, 256).to(self.device)
        
        # ============= Train Discriminator =============
        for discriminator in self.discriminators:
            discriminator.train()
        
        # Generate fake images
        noise = torch.randn(batch_size, 100).to(self.device)
        
        with torch.no_grad():
            fake_images, _, attn_maps = self.generator(
                noise, sent_embs, word_embs
            )
        
        # Discriminate real images
        real_logits_list, real_text_logits_list, _ = self.discriminators[0](
            real_images, sent_embs
        )
        
        # Discriminate fake images
        fake_logits_list, fake_text_logits_list, _ = self.discriminators[0](
            fake_images.detach(), sent_embs
        )
        
        # Discriminator loss
        d_loss = 0
        for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
            d_loss += self.loss_fn.discriminator_loss(real_logits, fake_logits)
        
        # Text-image discrimination loss
        for real_text_logits, fake_text_logits in zip(
            real_text_logits_list, fake_text_logits_list
        ):
            d_loss += 0.5 * self.loss_fn.discriminator_loss(
                real_text_logits, fake_text_logits
            )
        
        # Update discriminator
        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()
        
        # ============= Train Generator =============
        self.generator.train()
        
        # Generate fake images
        noise = torch.randn(batch_size, 100).to(self.device)
        fake_images, _, attn_maps = self.generator(
            noise, sent_embs, word_embs
        )
        
        # Discriminate fake images
        fake_logits_list, fake_text_logits_list, _ = self.discriminators[0](
            fake_images, sent_embs
        )
        
        # Generator loss
        g_loss = 0
        for fake_logits in fake_logits_list:
            g_loss += self.loss_fn.generator_loss(
                fake_logits, fake_images, real_images, attn_maps
            )
        
        # Text-image alignment loss
        for fake_text_logits in fake_text_logits_list:
            g_loss += 0.5 * (-fake_text_logits.mean())
        
        # Update generator
        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'loss_d': d_loss.item(),
            'loss_g': g_loss.item()
        }
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            avg_metrics: Average metrics for epoch
        """
        epoch_metrics = {
            'loss_d': [],
            'loss_g': []
        }
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            
            for key, val in metrics.items():
                epoch_metrics[key].append(val)
            
            if batch_idx % self.config.log_interval == 0:
                avg_d = sum(epoch_metrics['loss_d']) / len(epoch_metrics['loss_d'])
                avg_g = sum(epoch_metrics['loss_g']) / len(epoch_metrics['loss_g'])
                pbar.set_postfix({
                    'loss_d': f'{avg_d:.4f}',
                    'loss_g': f'{avg_g:.4f}'
                })
        
        # Average metrics
        avg_metrics = {}
        for key in epoch_metrics:
            avg_metrics[key] = sum(epoch_metrics[key]) / len(epoch_metrics[key])
        
        return avg_metrics
    
    def train(self, train_dataloader, num_epochs=None):
        """
        Full training loop
        
        Args:
            train_dataloader: Training dataloader
            num_epochs: Number of epochs (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            epoch_metrics = self.train_epoch(train_dataloader)
            
            # Log metrics
            for key, val in epoch_metrics.items():
                self.metrics[key].append(val)
            
            print(f"  Loss D: {epoch_metrics['loss_d']:.4f}")
            print(f"  Loss G: {epoch_metrics['loss_g']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminators[0].state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'metrics': self.metrics,
            'config': self.config.to_dict()
        }
        
        path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch:03d}.pt'
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminators[0].load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.metrics = checkpoint['metrics']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']


def create_trainer(
    generator,
    discriminators,
    text_encoder,
    loss_fn,
    config=None
):
    """
    Create trainer instance
    
    Args:
        generator: Generator model
        discriminators: List of discriminators
        text_encoder: Text encoder
        loss_fn: Loss function
        config: Training config
        
    Returns:
        Trainer instance
    """
    if config is None:
        config = TrainingConfig()
    
    device = config.device
    
    trainer = Trainer(
        generator=generator,
        discriminators=discriminators,
        text_encoder=text_encoder,
        loss_fn=loss_fn,
        device=device,
        config=config
    )
    
    return trainer
