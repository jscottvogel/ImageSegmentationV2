# Reusable Architecture Patterns & Design Guide

This guide compiles the key architectural patterns, training strategies, and optimization designs developed for the **FloodNet Synergistic Attention Network**. These patterns are modular and can be reused in any computer vision, multi-model ensemble, or high-precision semantic segmentation task.

---

## 1. Heterogeneous Feature-Level Fusion (Synergistic Fusion)

Instead of traditional ensemble methods like output logit averaging (soft voting), this pattern performs **feature-level fusion** by extracting and concatenating intermediate feature maps from diverse models.

```
       [Model A Decoder] --> (32 ch, 1/1 scale)  ---\
       [Model B Decoder] --> (64 ch, 1/4 scale)  ---> [Upsample & Projection] ---> [Concat (160 ch)] ---> [Fusion Head]
       [Model C Decoder] --> (512 ch, 1/8 scale) ---/
```

### Key Design Rules:
1. **Resolution Alignment**: Interpolate all low-resolution feature maps (e.g., ASPP or ResNet bottlenecks) back to $1/1$ resolution using bilinear upsampling.
2. **Channel Projection**: Apply $1 \times 1$ convolutions followed by BatchNorm and ReLU to reduce high-channel layers (e.g., FCN's 512 channels) to a standard channel size (e.g., 64). This prevents high-rank feature maps from dominating the concatenated representation.
3. **Diversity of Representations**: Fuse models with different architectural priors (e.g., UNet for precise borders, DeepLab for multi-scale context, FCN for robust localized features).

---

## 2. Sequential Dual-Attention (CBAM Block)

Fusing multi-model features creates a high-dimensional stack ($160$ channels) containing redundant information. The **CBAM (Convolutional Block Attention Module)** pattern sequentially solves the "what" and "where" attention problems.

```
Feature Stack (B, C, H, W) 
   │
   ├──> Channel Attention (AvgPool & MaxPool -> MLP -> Sigmoid) ──> Weight Channels (Select "Which Model")
   │
   └──> Spatial Attention (AvgPool & MaxPool -> 7x7 Conv -> Sigmoid) ──> Weight Pixels (Locate "Where Classes Are")
```

### Key Implementation:
```python
# Channel Attention: Pools spatially to compute channel-specific scales.
self.avg_pool = nn.AdaptiveAvgPool2d(1)
self.max_pool = nn.AdaptiveMaxPool2d(1)
self.fc = nn.Sequential(
    nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
)

# Spatial Attention: Pools channels to compute spatial-specific weight maps.
self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
```

---

## 3. Frozen-Decoder Fusion Fine-Tuning

When combining pre-trained, high-performance base segmentors, fine-tuning the entire network often causes **representation drift** and catastrophic forgetting. 

### Key Strategy:
1. **Unfreeze only the Fusion Head**: Set `requires_grad = False` for all backbone and decoder layers of the base models.
2. **High Learning Rate for Fusion**: Train the fusion head and attention modules with a moderate learning rate (e.g., $5 \times 10^{-4}$).
3. **Low/Zero Learning Rate for Decoders**: If decoders are unfrozen, train them at $2 \times 10^{-5}$ or lower.
4. **Rapid Convergence**: This approach converges in **3 epochs** rather than 15+, saving significant compute and avoiding overfitting.

---

## 4. Metric-Aligned Post-Processing Optimization

Many Kaggle and production segmentation metrics evaluate class-wise macro averages or row-wise Jaccard scores. The standard pixel-wise argmax over predictions is suboptimal for these metrics.

### Key Strategy:
1. **Multiclass Threshold Fallback**:
   - Assign probability thresholds class-by-class.
   - If a predicted pixel falls below the class threshold, reassign it to the highest-probability class among a set of *unconstrained fallback classes* (classes with $t=0$).
2. **Class-Specific Morphological Suppression**:
   - Rather than applying a single global size threshold (e.g., removing all blobs smaller than 96 pixels), sweep and assign morphological thresholds for each class independently.
   - For example, rare/tiny classes like **Pools** and **Vehicles** should have low/zero suppression thresholds, while background classes can have large thresholds.
3. **Decoupled Search**:
   - Optimize continuous parameters (weights, probabilities) first using **Powell's optimization**.
   - Optimize discrete parameters (morphological areas) second using **Greedy Coordinate Descent** on the entire validation set.
