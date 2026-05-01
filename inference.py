"""
Inference module for Res_AttnGAN
Generate images from text descriptions
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ResAttnGANInference:
    """
    Inference pipeline for Res_AttnGAN
    Generates images from text descriptions
    """
    
    def __init__(self, generator, text_encoder, device='cuda'):
        """
        Args:
            generator: Generator model
            text_encoder: Text encoder
            device: Device to use ('cuda' or 'cpu')
        """
        self.generator = generator.eval().to(device)
        self.text_encoder = text_encoder.eval().to(device)
        self.device = device
        
        # Freeze parameters
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def generate_images(self, captions, num_samples=1, return_attention=False):
        """
        Generate images from text captions
        
        Args:
            captions: List of text captions
            num_samples: Number of samples per caption
            return_attention: Whether to return attention maps
            
        Returns:
            images: Generated images (PIL Image objects)
            attention_maps: (Optional) Attention maps
        """
        batch_size = len(captions) * num_samples
        
        # Simple text encoding (in real scenario, use proper tokenizer)
        # For demo, create dummy embeddings
        sent_embs = torch.randn(batch_size, 256).to(self.device)
        word_embs = torch.randn(batch_size, 20, 256).to(self.device)
        
        # Generate noise
        noise = torch.randn(batch_size, 100).to(self.device)
        
        # Generate images
        fake_images, _, attn_maps = self.generator(
            noise, sent_embs, word_embs
        )
        
        # Convert to PIL images
        images = self._tensor_to_pil(fake_images)
        
        if return_attention:
            return images, attn_maps
        else:
            return images
    
    def _tensor_to_pil(self, tensor):
        """
        Convert tensor to PIL images
        
        Args:
            tensor: (batch_size, 3, height, width) normalized to [-1, 1]
            
        Returns:
            images: List of PIL Image objects
        """
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        
        # Clamp to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        images_np = tensor.cpu().numpy()
        
        # Convert to PIL
        images = []
        for img_np in images_np:
            # (3, height, width) -> (height, width, 3)
            img_np = np.transpose(img_np, (1, 2, 0))
            
            # Scale to [0, 255]
            img_np = (img_np * 255).astype(np.uint8)
            
            # Create PIL image
            img_pil = Image.fromarray(img_np)
            images.append(img_pil)
        
        return images
    
    def save_images(self, images, save_dir='outputs'):
        """
        Save generated images to disk
        
        Args:
            images: List of PIL Image objects
            save_dir: Directory to save images
        """
        Path(save_dir).mkdir(exist_ok=True)
        
        for i, img in enumerate(images):
            filename = f'{i:04d}.png'
            filepath = Path(save_dir) / filename
            img.save(filepath)
            print(f"Saved to {filepath}")
    
    def visualize_generation_process(
        self,
        caption,
        num_stages=4,
        figsize=(12, 3)
    ):
        """
        Visualize image generation across stages
        
        Args:
            caption: Text caption
            num_stages: Number of generation stages
            figsize: Figure size for plot
        """
        # Dummy embeddings
        sent_embs = torch.randn(1, 256).to(self.device)
        word_embs = torch.randn(1, 20, 256).to(self.device)
        noise = torch.randn(1, 100).to(self.device)
        
        # Generate images at each stage
        _, images_stages, attn_maps = self.generator(
            noise, sent_embs, word_embs
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, num_stages + 1, figsize=figsize)
        
        for i, img_tensor in enumerate(images_stages[:num_stages]):
            # Convert to PIL
            img_pil = self._tensor_to_pil(img_tensor)[0]
            axes[i].imshow(img_pil)
            axes[i].set_title(f'Stage {i+1}')
            axes[i].axis('off')
        
        # Final image
        final_img_pil = self._tensor_to_pil(images_stages[-1])[0]
        axes[-1].imshow(final_img_pil)
        axes[-1].set_title('Final')
        axes[-1].axis('off')
        
        plt.suptitle(f'Caption: {caption}')
        plt.tight_layout()
        
        return fig
    
    def visualize_attention_maps(
        self,
        caption,
        stage_idx=3,
        word_idx=0
    ):
        """
        Visualize attention maps
        
        Args:
            caption: Text caption
            stage_idx: Stage to visualize
            word_idx: Word index in caption
        """
        # Dummy embeddings
        sent_embs = torch.randn(1, 256).to(self.device)
        word_embs = torch.randn(1, 20, 256).to(self.device)
        noise = torch.randn(1, 100).to(self.device)
        
        # Generate
        _, images_stages, attn_maps = self.generator(
            noise, sent_embs, word_embs
        )
        
        # Get attention map
        attn_map = attn_maps[stage_idx]  # (1, seq_len, height, width)
        attn_map = attn_map[0, word_idx]  # (height, width)
        
        # Get generated image
        img_tensor = images_stages[stage_idx]
        img_pil = self._tensor_to_pil(img_tensor)[0]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].imshow(img_pil)
        axes[0].set_title('Generated Image')
        axes[0].axis('off')
        
        attn_np = attn_map.cpu().numpy()
        axes[1].imshow(attn_np, cmap='hot')
        axes[1].set_title(f'Attention Map (Stage {stage_idx+1})')
        axes[1].axis('off')
        
        plt.suptitle(f'Caption: {caption}')
        plt.tight_layout()
        
        return fig


class InteractiveGenerator:
    """
    Interactive image generation with different text prompts
    """
    
    def __init__(self, inference_pipeline):
        """
        Args:
            inference_pipeline: ResAttnGANInference instance
        """
        self.inference = inference_pipeline
    
    def batch_generate(self, captions_list, num_samples_per_caption=3):
        """
        Generate images for multiple captions
        
        Args:
            captions_list: List of text captions
            num_samples_per_caption: Number of samples per caption
            
        Returns:
            results: Dict mapping captions to generated images
        """
        results = {}
        
        for caption in captions_list:
            images = self.inference.generate_images(
                [caption],
                num_samples=num_samples_per_caption
            )
            results[caption] = images
        
        return results
    
    def visualize_batch(self, results, figsize=(15, 10)):
        """
        Visualize batch of generated images
        
        Args:
            results: Dict of caption -> images
            figsize: Figure size
        """
        num_captions = len(results)
        num_samples = len(list(results.values())[0])
        
        fig, axes = plt.subplots(
            num_captions,
            num_samples,
            figsize=figsize
        )
        
        if num_captions == 1:
            axes = axes.reshape(1, -1)
        
        for i, (caption, images) in enumerate(results.items()):
            for j, img in enumerate(images):
                axes[i, j].imshow(img)
                if j == 0:
                    axes[i, j].set_ylabel(caption[:30], fontsize=8)
                axes[i, j].axis('off')
        
        plt.tight_layout()
        return fig
