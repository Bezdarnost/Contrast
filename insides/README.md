Baseline is a HAT with (S)W-MSA replaced by SS2D blocks from VMambaV2 

<details>
<summary>Contrast baseline initialization:</summary>
  
```
model = Contrast(
        img_range=1., resi_connection='1conv', window_size=16, overlap_ratio=0.5,
        depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
        patch_size=1, in_chans=3, num_out_ch=3, dims=60, upscale_dims=48,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False,
        ssm_init="v2", forward_type="v05_noz", 
        mlp_ratio=2.0, mlp_act_layer="gelu", gmlp=False,
        patch_norm=True, norm_layer=nn.LayerNorm,
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, img_size=64, 
        upsampler='pixelshuffledirect', upscale=4, channel_first=False
    )
```
  
</details>

I've made the comparison in training with batch size 8 and patch size 64 of low-resolution image for X4 task on my RTX 3060:

<details>
<summary>HAT-light hyperparameters details:</summary>
  
```
Same depth, num_head, window_size, dims, upscaler and other details as my baseline Contrast
```
  
</details>

**Speed of training**
- HAT-light for one epoch (11,056 batches) took approximately 2 hours and 1 minute (**121 minutes**). 
- Baseline for one epoch (11,056 batches) took around 1 hour and 10 minutes (**70 minutes**). (The baseline is approximately **41.3% faster**).

**Memory requierements**
- HAT-light **11.2 GB**. 
- Baseline **8 GB.** (The baseline is approximately **28.6% better**).

I've made the training for 50 000 iterations and saved model weights and metrics for each 5 000 iterations, trained with MSELoss and Adam(lr=2e-4)

<p align="center">
  <img src="../images/baseline_vs_hat.png" width="50%">
</p>
