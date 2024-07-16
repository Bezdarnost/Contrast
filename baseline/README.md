I recommend:
- Python=3.11
- Torch=2.2.0
- Triton=2.2.0

You need to copy this repo:
https://github.com/MzeroMiko/VMamba

and build Mamba Kernels. It can be done this this code:

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba

conda create -n contrast
conda activate contrast

pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```

This model is a hybrid of [VMamba](https://github.com/MzeroMiko/VMamba) and [HAT](https://github.com/XPixelGroup/HAT)

Model Initialization:

```Python
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
