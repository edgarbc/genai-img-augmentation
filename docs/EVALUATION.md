# Evaluation Guide for Generative Image Augmentation

This document provides guidance on evaluating the quality and effectiveness of generative image augmentation, with special considerations for medical imaging applications.

## Overview

Evaluating generative models requires multiple complementary approaches:
1. **Image Quality Metrics**: Quantitative measures of generated image quality
2. **Distribution Metrics**: How well generated images match real data distribution
3. **Downstream Task Evaluation**: Impact on actual machine learning tasks
4. **Expert Evaluation**: Domain expert assessment (critical for medical imaging)

---

## Image Quality Metrics

### SSIM (Structural Similarity Index)

SSIM measures the structural similarity between two images, considering luminance, contrast, and structure.

```python
from skimage.metrics import structural_similarity as ssim

score = ssim(original_image, augmented_image)
# Score ranges from -1 to 1, with 1 being identical
```

**Interpretation**:
- SSIM > 0.9: Very high similarity (augmentation preserves structure)
- SSIM 0.7-0.9: Good similarity (moderate augmentation)
- SSIM < 0.7: Significant transformation

### PSNR (Peak Signal-to-Noise Ratio)

PSNR measures the ratio between maximum signal power and noise power.

```python
from skimage.metrics import peak_signal_noise_ratio as psnr

score = psnr(original_image, augmented_image)
# Higher values indicate better quality
```

**Interpretation**:
- PSNR > 40 dB: Excellent quality
- PSNR 30-40 dB: Good quality
- PSNR < 30 dB: Noticeable degradation

---

## Distribution Metrics

### FID (Fréchet Inception Distance)

FID measures the distance between feature distributions of real and generated images.

```python
# Requires torch and torchvision
# pip install torch torchvision pytorch-fid

# Command line:
# python -m pytorch_fid path/to/real path/to/generated
```

**Interpretation**:
- FID < 50: Good quality generation
- FID 50-100: Moderate quality
- FID > 100: Poor distribution match

### IS (Inception Score)

Measures both quality and diversity of generated images.

**Note**: IS is less suitable for specialized domains like medical imaging where ImageNet-pretrained features may not be appropriate.

---

## Downstream Task Evaluation

The ultimate test of augmentation effectiveness is improvement in downstream tasks.

### Recommended Protocol

1. **Baseline**: Train model on real data only, evaluate on held-out test set
2. **Augmented**: Train model on real + augmented data, evaluate on same test set
3. **Compare**: Measure improvement in task-specific metrics

### Example Evaluation Setup

```python
# Pseudocode for downstream evaluation

# 1. Split real data
train_real, val_real, test_real = split_data(real_images)

# 2. Generate augmented data
augmented_images = generate_augmented(train_real)

# 3. Train baseline model
baseline_model = train(train_real)
baseline_accuracy = evaluate(baseline_model, test_real)

# 4. Train augmented model
augmented_model = train(train_real + augmented_images)
augmented_accuracy = evaluate(augmented_model, test_real)

# 5. Compare
improvement = augmented_accuracy - baseline_accuracy
```

### Metrics by Task Type

| Task | Primary Metrics |
|------|----------------|
| Classification | Accuracy, F1-Score, AUC-ROC |
| Detection | mAP, IoU, Precision/Recall |
| Segmentation | Dice Score, IoU, Hausdorff Distance |

---

## Medical Imaging Specific Considerations

### ⚠️ Critical Safety Guidelines

1. **Never use synthetic data alone** for clinical validation
2. **Always involve domain experts** in evaluation
3. **Document all synthetic data** used in training
4. **Test on real clinical data** before any deployment

### Recommended Medical Evaluation Protocol

#### Phase 1: Technical Validation
1. Compute standard metrics (SSIM, FID)
2. Verify no training data memorization
3. Check for anatomical plausibility

#### Phase 2: Expert Review
1. Have radiologists/clinicians review generated samples
2. Identify any anatomical errors or artifacts
3. Assess clinical relevance of variations

#### Phase 3: Downstream Validation
1. Train models with and without augmentation
2. Validate on held-out real clinical data
3. Perform error analysis on failure cases

#### Phase 4: Clinical Considerations
1. Document limitations clearly
2. Obtain appropriate approvals (IRB, etc.)
3. Plan for ongoing monitoring

### Anatomical Plausibility Checklist

For medical images, verify:
- [ ] Correct anatomical structures present
- [ ] Appropriate relative sizes and positions
- [ ] No impossible anatomical configurations
- [ ] Realistic tissue textures and contrasts
- [ ] No obvious artifacts or noise patterns

---

## Evaluation Pitfalls to Avoid

### Common Mistakes

1. **Overfitting to metrics**: Optimizing only for FID may not improve downstream tasks
2. **Ignoring diversity**: Low FID with low diversity is not useful
3. **Data leakage**: Ensure test data is never seen during training or generation
4. **Domain shift**: Metrics on synthetic data may not reflect real-world performance

### Medical-Specific Pitfalls

1. **Using general-purpose metrics**: ImageNet features may not apply to medical images
2. **Ignoring class balance**: Medical datasets are often heavily imbalanced
3. **Assuming transferability**: Models trained on one scanner/hospital may not generalize
4. **Neglecting edge cases**: Rare pathologies need special attention

---

## Quick Reference

### Minimum Evaluation Checklist

- [ ] SSIM score computed and reasonable
- [ ] Visual inspection of samples (no obvious artifacts)
- [ ] Downstream task shows improvement (or at least no degradation)
- [ ] Domain expert review (for medical applications)

### Extended Evaluation Checklist

- [ ] FID score computed and compared to baselines
- [ ] Diversity analysis of generated samples
- [ ] Multiple downstream tasks evaluated
- [ ] Ablation studies (different augmentation amounts)
- [ ] Error analysis on failure cases
- [ ] Documentation of all data sources and preprocessing

---

## Resources

### Tools
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid): FID computation
- [scikit-image](https://scikit-image.org/): SSIM, PSNR, and other metrics
- [MONAI](https://monai.io/): Medical imaging deep learning framework

### References
- Borji, A. (2022). Pros and cons of GAN evaluation measures: New developments.
- Heusel, M., et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.
- Wang, Z., et al. (2004). Image quality assessment: from error visibility to structural similarity.
