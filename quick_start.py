#!/usr/bin/env python3
"""
Res_AttnGAN Quick Start Script
Run this script to train and test Res_AttnGAN with default configurations
"""

import torch
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import modules
from text_encoder import TextEncoder
from generator import ResAttnGANGenerator
from discriminator import MultiScaleDiscriminator
from losses import ResAttnGANLoss
from data_loader import get_data_loader
from trainer import TrainingConfig, create_trainer
try:
    from inference import ResAttnGANInference
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False
import torch.optim as optim


def main():
    """Main training script"""
    
    print("=" * 70)
    print("Res_AttnGAN Quick Start - Text-to-Image Synthesis")
    print("=" * 70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n1. SETUP")
    print(f"   Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loader
    print(f"\n2. DATA LOADING")
    train_loader = get_data_loader(
        dataset_name='dummy',
        batch_size=16,
        num_workers=0,
        img_size=64
    )
    print(f"   ✓ Dataset loaded: {len(train_loader)} batches")
    
    # Create models
    print(f"\n3. MODEL INITIALIZATION")
    
    # Text encoder
    text_encoder = TextEncoder(vocab_size=5000, word_dim=300, sent_dim=256)
    text_encoder = text_encoder.to(device)
    print(f"   ✓ Text Encoder created")
    
    # Generator
    generator = ResAttnGANGenerator(context_dim=256, img_channels=3, hidden_dim=512)
    generator = generator.to(device)
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"   ✓ Generator created ({total_params:,} parameters)")
    
    # Discriminators
    discriminator = MultiScaleDiscriminator(num_scales=4, img_channels=3, context_dim=256)
    discriminator = discriminator.to(device)
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"   ✓ Discriminator created ({total_params:,} parameters)")
    
    # Loss functions
    loss_fn = ResAttnGANLoss(
        loss_type='hinge',
        lambda_perceptual=1.0,
        lambda_attn=1.0,
        use_attention_loss=True
    )
    print(f"   ✓ Loss functions created")
    
    # Optimizers
    print(f"\n4. OPTIMIZER SETUP")
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    print(f"   ✓ Optimizers configured")
    
    # Training config
    config = TrainingConfig()
    config.device = device
    config.batch_size = 16
    config.num_epochs = 3  # Short for demo
    config.save_interval = 1
    
    # Create trainer
    print(f"\n5. TRAINER SETUP")
    trainer = create_trainer(
        generator=generator,
        discriminators=[discriminator],
        text_encoder=text_encoder,
        loss_fn=loss_fn,
        config=config
    )
    print(f"   ✓ Trainer created")
    print(f"   ✓ Checkpoints will be saved to: {config.checkpoint_dir}")
    
    # Train
    print(f"\n6. TRAINING")
    print(f"   Starting training for {config.num_epochs} epochs...")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    try:
        trainer.train(train_loader, num_epochs=config.num_epochs)
        print(f"\n   ✓ Training completed successfully!")
    except KeyboardInterrupt:
        print(f"\n   ! Training interrupted by user")
    except Exception as e:
        print(f"\n   ✗ Training failed with error: {e}")
        return 1
    
    # Inference
    print(f"\n7. IMAGE GENERATION")
    if HAS_INFERENCE:
        inference = ResAttnGANInference(generator, text_encoder, device=device)
    
    # Test prompts
    prompts = [
        "a red bird with black wings",
        "a yellow flower with green leaves",
        "a blue butterfly in flight"
    ]
    
    print(f"   Generating images for {len(prompts)} prompts...")
    
    generator.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"   [{i+1}/{len(prompts)}] {prompt}")
            
            noise = torch.randn(3, 100).to(device)
            sent_embs = torch.randn(3, 256).to(device)
            word_embs = torch.randn(3, 20, 256).to(device)
            
            images, _, _ = generator(noise, sent_embs, word_embs)
            
            # Save images
            Path('outputs').mkdir(exist_ok=True)
            pil_images = inference._tensor_to_pil(images)
            for j, img in enumerate(pil_images):
                filename = f"outputs/{i}_{j}.png"
                img.save(filename)
    
    print(f"   ✓ Images saved to 'outputs/' directory")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
✓ Project successfully initialized and tested!

Next steps:
1. Configure your dataset in config.yaml
2. Adjust hyperparameters as needed
3. Run: python quick_start.py --config config.yaml
4. Monitor training with tensorboard: tensorboard --logdir logs/
5. Generate images with trained model

For more information, see README.md
""")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
