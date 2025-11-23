---

# ML815 Distributed Training Experiment Files

This directory contains all configuration files, DeepSpeed settings, training scripts, and memory logs used in the ML815 project. The goal is to compare multiple distributed training strategies applied to the Qwen2.5-Coder-1.5B model using LLaMA-Factory, DeepSpeed, and optimized ROCm training on multi-GPU nodes.

---

## 1. File Types

### Training Configuration Files (`ml815_model*.yaml`)

Each YAML file defines a specific training experiment, including:

* Base model and data source
* Hyperparameters
* FlashAttention, Liger kernels, gradient checkpointing
* Optional DeepSpeed configuration
* Logging, checkpointing, and HuggingFace Hub metadata

Each `ml815_modelX.yaml` corresponds to one experimental training run.

---

### DeepSpeed Configuration Files (`ds_*.json`)

These JSON files describe the DeepSpeed engine settings used in different experiments.

| File                        | Description                                                   |
| --------------------------- | ------------------------------------------------------------- |
| `ds_z1_config.json`         | ZeRO-1 (optimizer state partitioning)                         |
| `ds_z2_config.json`         | ZeRO-2 (optimizer + gradient sharding)                        |
| `ds_z2_compress.json`       | ZeRO-2 with 8-bit gradient compression                        |
| `ds_z2_offload_config.json` | ZeRO-2 with CPU offloading of parameters and optimizer states |

These configurations significantly affect performance, memory overhead, and communication cost.

---

### Training Launch Scripts (`train_model*.sh`)

Each shell script activates the ROCm environment, sets multi-GPU distributed variables, launches LLaMA-Factory via `torchrun`, and logs GPU VRAM usage every five minutes.
The scripts map one-to-one with the corresponding YAML files.

---

### GPU Memory Logs (`gpu_mem_model*.log`)

These logs contain periodic VRAM readings from `rocm-smi`.
They are used to compare memory usage across training strategies.

---

## 2. Overview of the Eight Model Variants

Eight training runs were executed using different distributed training strategies and DeepSpeed optimizations.
Only six of these represent distinct methods; one is a baseline and one is redundant (ZeRO-0).

| Model       | Config File         | Technique                                | Notes                                                                      |
| ----------- | ------------------- | ---------------------------------------- | -------------------------------------------------------------------------- |
| **Model 1** | `ml815_model1.yaml` | Baseline Distributed Data Parallel (DDP) | Standard multi-GPU training without DeepSpeed.                             |
| **Model 2** | `ml815_model2.yaml` | ZeRO-1                                   | Partitions optimizer states across GPUs.                                   |
| **Model 3** | `ml815_model3.yaml` | ZeRO-2                                   | Partitions optimizer states and gradients.                                 |
| **Model 4** | `ml815_model4.yaml` | Overlap-DDP                              | Uses communication/computation overlap to reduce synchronization stalls.   |
| **Model 5** | `ml815_model5.yaml` | High Gradient Accumulation               | Simulates large batch sizes via gradient accumulation.                     |
| **Model 6** | `ml815_model6.yaml` | ZeRO-0                                   | Functionally identical to baseline DDP; not included in results.           |
| **Model 7** | `ml815_model7.yaml` | ZeRO-2 with Gradient Compression         | Quantizes gradients during communication to reduce bandwidth.              |
| **Model 8** | `ml815_model8.yaml` | ZeRO-2 with CPU Offloading               | Moves parameters and optimizer states to CPU RAM, reducing GPU VRAM usage. |

Only **six models** provide distinct methodological comparisons (Models 1, 2, 3, 4, 5, 7, and 8).

---

If you'd like, I can also generate the remaining sections (slides overview, diagram-ready descriptions of each method, or a more detailed methodology section) in the same clean public style.
