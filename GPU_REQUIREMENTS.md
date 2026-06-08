# GPU Training Requirements

This document outlines the necessary steps to transition the model training pipeline from the CPU-based prototype to an NVIDIA GPU-enabled environment.

## 1. Hardware Prerequisites
* **GPU:** NVIDIA GPU with at least 8GB+ VRAM (12GB+ recommended for `seq_len=256` and larger batch sizes).
* **Driver:** Latest NVIDIA proprietary drivers installed.

## 2. Software Stack
* **CUDA Toolkit:** Matches your driver version (v12.x recommended).
* **PyTorch with CUDA:** Re-install PyTorch to ensure it is the CUDA-enabled version rather than the CPU-only version.
    ```bash
    # Example for CUDA 12.4
    pip uninstall torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```

## 3. Environment & Dependencies
* Re-create the virtual environment or update it.
* Ensure `opencv-python` and `av` are installed for video processing.
* Verify CUDA availability in Python:
    ```python
    import torch
    print(torch.cuda.is_available()) # Must return True
    ```

## 4. Configuration Adjustments for GPU Training
When moving to GPU, you should increase the `batch_size` to take advantage of parallel processing:

* **Batch Size:** Increase from `1` (CPU) to `4`, `8`, or `16` (depending on your VRAM).
* **Learning Rate:** You may need to scale your learning rate. A common rule of thumb is to increase the LR when increasing the batch size (e.g., if you quadruple your batch size, you can often double your LR).
* **Num Workers:** In `train.py`, you can increase `num_workers` in the `DataLoader` to speed up video frame fetching:
    ```python
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    ```

## 5. Performance Monitoring
* Use `nvidia-smi` to monitor GPU VRAM usage during the first epoch.
* Ensure you are logging to `mlflow` to track if the higher batch size improves training stability (smoothing out the loss curve).
