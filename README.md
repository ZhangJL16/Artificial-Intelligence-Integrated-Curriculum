# DCGAN
  
  python DCGAN_1.py

# Diffusion

  python diffusion.py

# DiT
train
  torchrun --nnodes=1 --nproc_per_node=1 train.py --model DiT-XL/2 --data-path data 
sample
  python sample.py --model DiT-XL/2 --image-size 256 --ckpt /your_ckpt
