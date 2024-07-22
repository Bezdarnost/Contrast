I recommend:
- Python=3.11
- Torch=2.3.0
- Triton=2.3.0

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

and you also can install other libraries from requirements

# Training:

```bash
# Contrast-light, input=64x64, 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_light_x4.yml --launcher pytorch

# HAT-light(pure transformer)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_HAT_light_x4.yml --launcher pytorch

# Mamba-light(pure mamba)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Mamba_light_x4.yml --launcher pytorch

# Contrast-light without PE
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_light_no_pe_x4.yml --launcher pytorch

# Contrast-light without PE and without CAB
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_light_no_pe_no_cab_x4.yml --launcher pytorch
```
