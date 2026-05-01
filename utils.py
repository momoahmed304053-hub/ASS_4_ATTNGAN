"""
Utility functions for visualization and analysis
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ModelAnalyzer:
    """Analyze model architecture and performance"""
    
    @staticmethod
    def count_parameters(model):
        """Count total trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_summary(model):
        """Print detailed model summary"""
        total_params = ModelAnalyzer.count_parameters(model)
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        # Count layers
        num_layers = len(list(model.parameters()))
        print(f"Number of parameter groups: {num_layers}")
    
    @staticmethod
    def analyze_gradients(model):
        """Analyze gradient statistics"""
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                grad_norms.append(grad_norm.item())
        
        if grad_norms:
            return {
                'mean': np.mean(grad_norms),
                'std': np.std(grad_norms),
                'min': np.min(grad_norms),
                'max': np.max(grad_norms)
            }
        return None


class AttentionVisualizer:
    """Visualize attention maps"""
    
    @staticmethod
    def plot_attention_map(attn_map, image=None, word_idx=0, title="Attention Map"):
        """
        Plot attention map with optional background image
        
        Args:
            attn_map: (height, width) attention map
            image: (3, height, width) background image
            word_idx: Word index for labeling
            title: Plot title
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Plot attention map
        attn_np = attn_map.cpu().numpy() if isinstance(attn_map, torch.Tensor) else attn_map
        
        if image is not None:
            # Normalize image
            img_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np + 1) / 2  # Denormalize from [-1, 1]
            img_np = np.clip(img_np, 0, 1)
            
            # Show image
            ax.imshow(img_np, alpha=0.6)
        
        # Overlay attention
        im = ax.imshow(attn_np, cmap='jet', alpha=0.6)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        
        return fig
    
    @staticmethod
    def plot_attention_grid(attn_maps, image=None, num_words=5, figsize=(15, 3)):
        """
        Plot multiple attention maps in a grid
        
        Args:
            attn_maps: (seq_len, height, width) attention maps
            image: Background image
            num_words: Number of words to visualize
            figsize: Figure size
        """
        seq_len = attn_maps.size(0) if isinstance(attn_maps, torch.Tensor) else attn_maps.shape[0]
        num_to_plot = min(num_words, seq_len)
        
        fig, axes = plt.subplots(1, num_to_plot + 1, figsize=figsize)
        
        # Plot image
        if image is not None:
            img_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np + 1) / 2
            img_np = np.clip(img_np, 0, 1)
            axes[0].imshow(img_np)
            axes[0].set_title('Generated Image')
        axes[0].axis('off')
        
        # Plot attention maps
        attn_np = attn_maps.cpu().numpy() if isinstance(attn_maps, torch.Tensor) else attn_maps
        
        for i in range(num_to_plot):
            axes[i+1].imshow(attn_np[i], cmap='hot')
            axes[i+1].set_title(f'Word {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        return fig


class ImageQualityMetrics:
    """Compute image quality metrics"""
    
    @staticmethod
    def compute_brightness(images):
        """Compute average brightness"""
        return images.mean().item()
    
    @staticmethod
    def compute_contrast(images):
        """Compute image contrast"""
        return images.std().item()
    
    @staticmethod
    def compute_sharpness(images):
        """Estimate image sharpness using Laplacian"""
        from scipy import ndimage
        
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        sharpness_scores = []
        for img in images:
            # Convert to grayscale
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            # Apply Laplacian
            laplacian = ndimage.laplace(gray)
            sharpness = np.abs(laplacian).mean()
            sharpness_scores.append(sharpness)
        
        return np.mean(sharpness_scores)
    
    @staticmethod
    def compute_diversity(images):
        """Compute batch diversity using feature correlation"""
        if isinstance(images, torch.Tensor):
            images = images.view(images.size(0), -1)
        else:
            images = images.reshape(images.shape[0], -1)
        
        # Compute pairwise correlations
        correlations = np.corrcoef(images)
        
        # Return average off-diagonal correlation (lower = more diverse)
        mask = ~np.eye(correlations.shape[0], dtype=bool)
        return np.abs(correlations[mask]).mean()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict, title="Metrics Comparison"):
        """
        Plot metric comparison
        
        Args:
            metrics_dict: {metric_name: value} dict
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        names = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Metric Value', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig


class ComparisonVisualizer:
    """Visualize comparisons between models"""
    
    @staticmethod
    def compare_generations(images_dict, figsize=(18, 6)):
        """
        Compare generated images from different models
        
        Args:
            images_dict: {model_name: images} dict
            figsize: Figure size
        """
        num_models = len(images_dict)
        num_samples = list(images_dict.values())[0].size(0)
        
        fig, axes = plt.subplots(num_samples, num_models, figsize=figsize)
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for col, (model_name, images) in enumerate(images_dict.items()):
            images_display = (images.cpu() + 1) / 2
            images_display = torch.clamp(images_display, 0, 1)
            
            for row in range(num_samples):
                ax = axes[row, col]
                img_np = images_display[row].permute(1, 2, 0).numpy()
                ax.imshow(img_np)
                
                if row == 0:
                    ax.set_title(model_name, fontweight='bold', fontsize=12)
                ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def compare_metrics(metrics_dict, figsize=(14, 6)):
        """
        Compare metrics across models
        
        Args:
            metrics_dict: {model_name: {metric: value}} nested dict
            figsize: Figure size
        """
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [metrics_dict[model][metric] for model in models]
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
            bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig


class ExperimentLogger:
    """Log and save experiment results"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def save_images(self, images, name, epoch=None):
        """Save batch of images"""
        save_dir = self.log_dir / 'images'
        save_dir.mkdir(exist_ok=True)
        
        if epoch is not None:
            filename = f"{name}_epoch_{epoch:03d}.png"
        else:
            filename = f"{name}.png"
        
        # Convert to grid
        from torchvision.utils import make_grid
        grid = make_grid(images, nrow=4, normalize=True)
        
        from torchvision.transforms import ToPILImage
        pil_img = ToPILImage()(grid)
        pil_img.save(save_dir / filename)
        
        return save_dir / filename
    
    def save_metrics(self, metrics, name):
        """Save metrics to CSV"""
        import csv
        
        metrics_dir = self.log_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        filepath = metrics_dir / f"{name}.csv"
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())
        
        return filepath
    
    def save_config(self, config_dict, name='config'):
        """Save configuration"""
        import json
        
        config_dir = self.log_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        
        filepath = config_dir / f"{name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return filepath
