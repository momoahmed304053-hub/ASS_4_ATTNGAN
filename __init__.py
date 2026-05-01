#!/usr/bin/env python3
"""
Res_AttnGAN - Residual Spatial Attention GAN for Text-to-Image Synthesis
Implementation based on the paper: "Res_AttnGAN: Residual Spatial Attention GAN 
for Fine-grained Text to Image Synthesis"

This is an enhanced version of AttnGAN that addresses its drawbacks through:
1. Residual spatial attention mechanisms
2. Dense feature connections
3. Multi-scale residual blocks
4. Improved text-image alignment
"""

__version__ = "1.0.0"
__author__ = "Research Team"

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
