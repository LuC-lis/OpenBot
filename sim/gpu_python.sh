#!/usr/bin/env bash
# 用 openbot-gpu 环境跑 python，并设置 TF 2.15 所需的 CUDA 12 运行库路径。
#
# 背景：TF 2.15.1 需要 CUDA 12.2；pip 装的 nvidia-*-cu12 库在
#       site-packages/nvidia/*/lib 下，TF 不会自动去找，必须加进 LD_LIBRARY_PATH。
#
# 用法：
#   cd OpenBot/policy
#   ../sim/gpu_python.sh -m openbot.train --no_tf_record ...
set -e

PREFIX=/home/luc/miniconda3/envs/openbot-gpu
NVIDIA_LIBS=$(ls -d "$PREFIX"/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
# TF 用 XLA 给某些算子做 JIT 时需要 CUDA 的 libdevice.10.bc
# （它在 nvidia-cuda-nvcc-cu12 的 nvvm/libdevice 下）
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${PREFIX}/lib/python3.10/site-packages/nvidia/cuda_nvcc ${XLA_FLAGS}"

exec "$PREFIX/bin/python" "$@"
