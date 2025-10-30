#!/bin/bash
source ~/.bashrc
mamba activate isonet_noise
CUDA_VISIBLE_DEVICES=0 python /public/home/yuyibei2023/IsoNet-unet-noise/IsoNet/bin/gennoise.py
