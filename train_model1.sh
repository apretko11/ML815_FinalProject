#!/bin/bash
ENV_NAME="vllm_rocm"
export NNODES=1
export NPROC_PER_NODE=4
export HIP_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_gpu_IDS=0,1,2,3
export PATH=~/local_cuda/bin:$PATH
export LD_LIBRARY_PATH=~/local_cuda/lib64:$LD_LIBRARY_PATH
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29511

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate $ENV_NAME
export PATH="/share/miniconda3/envs/vllm_rocm/bin:$PATH"
cd LLaMA-Factory
export HF_TOKEN=""
export WANDB_API_KEY=""

# Start GPU memory logging in the background
LOGFILE="gpu_mem_model1.log"

(
  while true; do
    echo "===== $(date) =====" >> $LOGFILE
    rocm-smi --showmeminfo vram >> $LOGFILE
    echo "" >> $LOGFILE
    sleep 300   # 5 minutes
  done
) &
MONITOR_PID=$!

FORCE_TORCHRUN=1 llamafactory-cli train examples/ml815_model1.yaml

# Stop monitoring after training ends
kill $MONITOR_PID

