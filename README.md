# genai-img-augmentation

![GitHub](https://img.shields.io/github/license/edgarbc/genai-img-augmentation)
[![CI](https://github.com/edgarbc/genai-img-augmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/edgarbc/genai-img-augmentation/actions/workflows/ci.yml)

> **TL;DR**: A demonstration of generative AI techniques for augmenting medical imaging datasets, specifically panoramic dental X-rays, using MONAI's variational autoencoder framework.

![pan xray](https://github.com/edgarbc/genai-img-augmentation/blob/main/img/IMG_0010.png)

## Problem Statement

Deep learning has proven effective for medical image analysis [1], but model development is often limited by:
- **Data scarcity**: Limited access to labeled medical images
- **Privacy restrictions**: HIPAA/GDPR constraints on patient data
- **High labeling costs**: Expert annotation is expensive and time-consuming

**Solution**: Use generative AI to synthesize realistic medical images that augment limited real datasets, improving model robustness while respecting privacy constraints.

## What's Included

| File/Directory | Description |
|----------------|-------------|
| [`my_monai_panxray_autoencoder.ipynb`](my_monai_panxray_autoencoder.ipynb) | Full MONAI VAE training notebook |
| [`demo/run_augmentation_demo.py`](demo/run_augmentation_demo.py) | Lightweight augmentation pipeline demo |
| [`demo/sample_image.png`](demo/sample_image.png) | Synthetic 128×128 test image |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model documentation with safety guidelines |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Evaluation metrics guide (FID, SSIM, etc.) |

## Quickstart

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/edgarbc/genai-img-augmentation.git
cd genai-img-augmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python demo/run_augmentation_demo.py
```

Expected output:
```
============================================================
Medical Image Augmentation Demo
============================================================

Loading sample image from: demo/sample_image.png
Image shape: (128, 128)

Applying augmentations...
✓ Rotation (15°) saved to: demo/output/augmented_rotated.png
✓ Gaussian noise (σ=20) saved to: demo/output/augmented_noisy.png

Computing quality metrics...
SSIM (original vs rotated): 0.2423
SSIM (original vs noisy):   0.4616

Demo Complete!
```

### 3. Full MONAI Experiments (Optional)

For the complete generative model training with MONAI:

```bash
# Install additional dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install monai matplotlib tqdm

# Open the notebook
jupyter notebook my_monai_panxray_autoencoder.ipynb
```

> ⚠️ **Note**: Full MONAI experiments require a GPU for reasonable training times. The notebook was designed for Google Colab with T4 GPU.

## Expected Results

### Demo Output

The lightweight demo produces:
- `demo/output/augmented_rotated.png` - Image rotated by 15°
- `demo/output/augmented_noisy.png` - Image with Gaussian noise added
- SSIM metrics comparing augmented vs. original images

### Full VAE Training (Notebook)

The MONAI notebook trains a variational autoencoder that:
- Learns latent representations of panoramic X-rays
- Generates new synthetic X-ray images from the learned distribution
- Can interpolate between images in latent space

<!-- 
## Visual Results

![Sample Augmentation Results](docs/images/results_placeholder.png)
*Placeholder: Add sample augmented images here*
-->

## ⚠️ Medical Safety Notice

**This project is for educational and research purposes only.**

- ❌ **NOT** for clinical diagnosis or patient care
- ❌ **NOT** validated for production medical systems
- ❌ **NOT** a replacement for real, properly consented medical data
- ✅ Useful for understanding generative AI techniques
- ✅ Demonstrates pipeline architecture for medical imaging

**Before using in any medical context:**
1. Obtain appropriate IRB approval
2. Validate with domain experts (radiologists, clinicians)
3. Test extensively on real clinical data
4. Review the [MODEL_CARD.md](MODEL_CARD.md) for detailed limitations

## Data Provenance

| Dataset | Description | Status |
|---------|-------------|--------|
| Demo synthetic image | Programmatically generated 128×128 grayscale | ✅ No patient data |
| Notebook training data | User must provide their own | ⚠️ User responsibility |

The demo included in this repository uses **only synthetic data** generated programmatically. No real patient images are included.

## Project Structure

```
genai-img-augmentation/
├── demo/
│   ├── run_augmentation_demo.py    # Lightweight demo script
│   ├── sample_image.png            # Synthetic test image
│   └── output/                     # Generated outputs
├── docs/
│   └── EVALUATION.md               # Evaluation guide
├── img/
│   └── IMG_0010.png                # Sample X-ray image
├── .github/workflows/
│   └── ci.yml                      # CI smoke test
├── my_monai_panxray_autoencoder.ipynb
├── MODEL_CARD.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Notebooks

- **[Panoramic X-ray VAE Training](my_monai_panxray_autoencoder.ipynb)**: Full variational autoencoder implementation using MONAI. Includes data loading, model architecture, training loop, and image generation.

## Evaluation

See [docs/EVALUATION.md](docs/EVALUATION.md) for guidance on:
- Image quality metrics (SSIM, PSNR)
- Distribution metrics (FID)
- Downstream task evaluation
- Medical-specific considerations

## Blog Post

This repository accompanies a [Medium blog post](https://medium.com/@viajesubmarino/synthetic-x-ray-dataset-augmentation-using-generative-ai-178ebc15a074) explaining the concepts and implementation details.

## References

[1] Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*.

[2] Tsuneki, M. (2022). Deep learning models in medical image analysis. *Journal of Oral Biosciences*.

[3] Koohi-Moghadam, M., Bae, K.T. (2023). Generative AI in Medical Imaging: Applications, Challenges, and Ethics. *J Med Syst* 47, 94.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure any contributions:
1. Include appropriate documentation
2. Follow the medical safety guidelines
3. Do not include any real patient data
