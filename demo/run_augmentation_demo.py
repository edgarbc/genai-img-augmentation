#!/usr/bin/env python3
"""
Lightweight Augmentation Demo for Medical Image Augmentation Pipeline

This demo demonstrates the image augmentation pipeline without requiring
heavy dependencies like MONAI or GPU. It shows:
1. Loading a synthetic grayscale medical-style image
2. Applying simple augmentations (rotation, noise)
3. Computing quality metrics (SSIM)
4. Saving outputs

For full generative model experiments, see the MONAI notebook.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

# Try to import skimage metrics; gracefully handle if unavailable
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. SSIM metrics will be skipped.")


def create_synthetic_image(size: int = 128) -> np.ndarray:
    """
    Create a synthetic grayscale image that simulates a simple medical imaging pattern.
    
    This creates a basic pattern with:
    - Gradient background (simulating tissue density variation)
    - Circular structures (simulating anatomical features)
    - Some random variation
    
    Args:
        size: Image dimension (creates size x size image)
    
    Returns:
        Grayscale image as numpy array (0-255 uint8)
    """
    np.random.seed(42)  # For reproducibility
    
    # Create gradient background
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    background = (xx * 0.3 + yy * 0.3) * 255
    
    # Add circular structures (simulating anatomical features)
    center1 = (size // 3, size // 2)
    center2 = (2 * size // 3, size // 2)
    radius = size // 6
    
    for cx, cy in [center1, center2]:
        dist = np.sqrt((xx * size - cx) ** 2 + (yy * size - cy) ** 2)
        circle = np.where(dist < radius, 200, 0)
        background = np.maximum(background, circle)
    
    # Add subtle random variation
    noise = np.random.normal(0, 10, (size, size))
    image = np.clip(background + noise, 0, 255).astype(np.uint8)
    
    return image


def apply_rotation(image: np.ndarray, angle: float = 15.0) -> np.ndarray:
    """
    Apply rotation augmentation using PIL.
    
    Args:
        image: Input grayscale image as numpy array
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image as numpy array
    """
    pil_image = Image.fromarray(image)
    rotated = pil_image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    return np.array(rotated)


def apply_gaussian_noise(image: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """
    Apply Gaussian noise augmentation.
    
    Args:
        image: Input grayscale image as numpy array
        sigma: Standard deviation of Gaussian noise
    
    Returns:
        Noisy image as numpy array
    """
    np.random.seed(123)  # For reproducibility
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def compute_ssim(original: np.ndarray, augmented: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between original and augmented images.
    
    Args:
        original: Original image as numpy array
        augmented: Augmented image as numpy array
    
    Returns:
        SSIM score (float between -1 and 1, higher is more similar)
    """
    if not SKIMAGE_AVAILABLE:
        return float('nan')
    
    return ssim(original, augmented, data_range=255)


def main():
    """Run the augmentation demo pipeline."""
    print("=" * 60)
    print("Medical Image Augmentation Demo")
    print("=" * 60)
    print()
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    sample_image_path = script_dir / "sample_image.png"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Load or create sample image
    if sample_image_path.exists():
        print(f"Loading sample image from: {sample_image_path}")
        original = np.array(Image.open(sample_image_path).convert('L'))
    else:
        print("Creating synthetic sample image...")
        original = create_synthetic_image(128)
        # Save the synthetic image for future runs
        Image.fromarray(original).save(sample_image_path)
        print(f"Saved synthetic image to: {sample_image_path}")
    
    print(f"Image shape: {original.shape}")
    print(f"Image dtype: {original.dtype}")
    print(f"Pixel range: [{original.min()}, {original.max()}]")
    print()
    
    # Apply augmentations
    print("Applying augmentations...")
    print("-" * 40)
    
    # Rotation augmentation
    rotated = apply_rotation(original, angle=15.0)
    rotated_path = output_dir / "augmented_rotated.png"
    Image.fromarray(rotated).save(rotated_path)
    print(f"✓ Rotation (15°) saved to: {rotated_path}")
    
    # Noise augmentation
    noisy = apply_gaussian_noise(original, sigma=20.0)
    noisy_path = output_dir / "augmented_noisy.png"
    Image.fromarray(noisy).save(noisy_path)
    print(f"✓ Gaussian noise (σ=20) saved to: {noisy_path}")
    
    print()
    
    # Compute metrics
    print("Computing quality metrics...")
    print("-" * 40)
    
    ssim_rotated = compute_ssim(original, rotated)
    ssim_noisy = compute_ssim(original, noisy)
    
    print(f"SSIM (original vs rotated): {ssim_rotated:.4f}")
    print(f"SSIM (original vs noisy):   {ssim_noisy:.4f}")
    print()
    
    # Summary
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("This demo shows the basic pipeline structure:")
    print("  1. Load/create source image")
    print("  2. Apply augmentations (rotation, noise)")
    print("  3. Compute quality metrics (SSIM)")
    print("  4. Save outputs")
    print()
    print("For full generative augmentation with VAE/diffusion models,")
    print("see the MONAI notebook: my_monai_panxray_autoencoder.ipynb")
    print()
    print("⚠️  Medical Safety Note: This demo uses synthetic data only.")
    print("    Real medical applications require proper clinical validation.")
    
    return 0


if __name__ == "__main__":
    exit(main())
