# ContrastV2
This is an official implementation for "ContrastV2: Marrying Convolutions, Transformers and State Space for Image Resoration". 

# Plan

## Changes:

### Model Design

- VMambaV2
- VMambaV2 + Hybrid Attention
- Attention -> Flash Attention
- Absolute Position Encoder(APE)
- APE -> Rotary Positional Encoder(RoPE)
- Conv for corners
- Conv 1x1 -> RFB
- Conv 3x3 -> Conv 5x5
- +GRN (Normalization)
- Residual from the start to the end sum -> concatenate
- Mixture of Experts

### Data Augmentation

- RGB Channel Shuffle
- Rotation
- Horizontal and Vertical Flip

### More Data & Training Time

- DIV2K -> DF2K
- DF2K -> DF2K + LSDIR
- 500k iter -> 2.5m iter

