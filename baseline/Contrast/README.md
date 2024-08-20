# NEW TRAININGS

```bash
# Light version
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_baseline_v3_x4.yml  --launcher pytorch
```

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

# Previous trainings:
```bash
# Baseline training, 4 GPUs 4 batch size
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_baseline_x4.yml --launcher pytorch

# Contrast-light, input=64x64, 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_baseline_gated_convffn_no_gated_light_x4.yml --launcher pytorch

# HAT-light(pure transformer)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_baseline_gated_convffn_light_x4.yml --launcher pytorch
```

# Training with torchrun:

```bash
# Contrast-light, input=64x64, 4 GPUs
torchrun --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_light_x4.yml --launcher pytorch

# HAT-light(pure transformer)
torchrun --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_HAT_light_x4.yml --launcher pytorch

# Mamba-light(pure mamba)
torchrun --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Mamba_light_x4.yml --launcher pytorch

# Contrast-light without PE
torchrun --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_light_no_pe_x4.yml --launcher pytorch

# Contrast-light without PE and without CAB
torchrun --nproc_per_node=4 --master_port=4321 train.py -opt options/Train/train_Contrast_light_no_pe_no_cab_x4.yml --launcher pytorch
```

# Training with torch.distributed.launch:

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
