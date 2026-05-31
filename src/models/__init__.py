"""
Model architectures for NeuralRestore pipeline.

Exports:
    - MultiTaskDamageDetector : Model 1 — damage detection + classification
    - LaMaGenerator          : Model 2 — inpainting
    - DnCNN                  : Model 3 — denoising
    - NAFNet                 : Model 4 — deblurring
    - RealESRGAN             : Model 5 — super resolution
"""

from .damage_detector import MultiTaskDamageDetector
from .inpainting import LaMaGenerator
from .denoising import DnCNN
from .deblurring import NAFNet
from .super_resolution import RealESRGAN
