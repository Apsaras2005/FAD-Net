# FAD-Net: Frequency-Domain Amplitude-Phase Decoupling Network for Optical-Elevation Remote Sensing Segmentation

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TGRS-blue.svg)](https://ieeexplore.ieee.org/document/11482238)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-red.svg)](https://pytorch.org/)

**Official PyTorch implementation of the TGRS paper "FAD-Net: Frequency-Domain Amplitude-Phase Decoupling Network for Optical-Elevation Remote Sensing Segmentation"**

---

## 📢 Optimized Implementation

To ensure maximum efficiency and training stability, this repository incorporates both architectural and computational pipeline optimizations over the original implementation.

### 1. Data-Driven Dynamic Routing (FGDWM)
- **End-to-End Differentiable:** Spectral energy (mean absolute amplitude) is extracted directly from Gabor filters and processed via Log-compression, LayerNorm, and a compact MLP, allowing routing weights to update via backpropagation.
- **Overhead Reduction:** Replaced iterative clustering loops and pairwise distance computations (`torch.cdist`) with an end-to-end projection layer, reducing peak memory usage and latency.

### 2. High-Efficiency Frequency-Domain Computation
- **RFFT Real-Transform Formulation:** Replaced two-dimensional complex FFTs (`fft2`/`ifft2`) in the Decoder with conjugate-symmetric real FFTs (`rfft2`/`irfft2`), reducing spectral feature map memory footprint by approximately 50%.
- **Linearity-Driven IFFT Reduction:** Leveraged the linearity property of the Fourier transform in `MM_Mona`. The $K$ individual inverse transforms are merged into a single `ifft2` call applied to the weighted composite spectrum.
- **Elimination of Coordinate Shifts:** Pre-shifted Log-Gabor filter frequencies to operate directly on unshifted spectral representations, removing runtime `fftshift` and `ifftshift` data movement overhead.

### 3. Tensor Caching & Kernel Acceleration
- **Native SDPA Integration:** Switched window attention to `torch.nn.functional.scaled_dot_product_attention`, executing via FlashAttention or Memory-Efficient kernels based on hardware availability. Cross-attention layers set `need_weights=False` to eliminate attention matrix retention.
- **Reference Points & Coordinate Caching:** Static geometric coordinate grids, trigonometric tables, deformable attention reference points, and spatial shape tensors are computed once and stored in memory caches to eliminate repetitive host-device kernel invocations.

---

## 🛠️ Environment Setup

To run this project, you need a Python environment with PyTorch installed. It is recommended to use **Python 3.8+** and **PyTorch 2.10+**.

You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```


## 📂 Ensure your project directory matches the following structure:

```text
FAD-Net/
├── config.py
├── train.py
├── model/
├── datasets/
│   ├── Vaihingen/
│   │   ├── top/
│   │   ├── dsm/
│   │   └── ground_truth/
│   └── Potsdam/
│       ├── top/
│       ├── dsm/
│       └── ground_truth/
├── pretrained_weights/
├── model_weights/
└── logs/
```

---

## 💾 Supported Datasets: ISPRS Vaihingen and Potsdam.

1. **Download Data:**
Please refer to the **MMSegmentation Dataset Preparation Guide** to download the raw data:
👉 **[Download ISPRS Datasets Here](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)**
2. **Organize Data:**
After downloading, you must extract and rename/move the files to match the `FILE_TEMPLATES` in `config.py`.
* **Vaihingen:** Ensure images are named like `top_mosaic_09cm_area1.tif`.
* **Potsdam:** Ensure images are named like `top_potsdam_2_10_RGB.tif`.


*Note: The script expects specific subfolders (`top`, `dsm`, `ground_truth`). Please ensure your extracted files are moved into these folders inside `datasets/Vaihingen` or `datasets/Potsdam`.*

---

## 🏋️ Pretrained Weights & Configuration

👉 **[Download Swin-V2 Weights (Microsoft GitHub)](https://github.com/microsoft/Swin-Transformer)**

Download the Swin Transformer V2 weights and place them in the `pretrained_weights/` folder. Update `config.py` according to your selected backbone:

### Swin-V2 Base

```python
'PRETRAINED_WEIGHTS_PATH': 'pretrained_weights/swinv2_base_patch4_window12_192_22k.pth',

'SWINV2': {
    'EMBED_DIM': 128,              
    'DEPTHS': [2, 2, 18, 2],       
    'NUM_HEADS': [4, 8, 16, 32],   
    'WINDOW_SIZE': 16,             
    'PRETRAINED_WINDOW_SIZES': [12, 12, 12, 6], 
}

```

### Swin-V2 Small

```python
'PRETRAINED_WEIGHTS_PATH': 'pretrained_weights/swinv2_small_patch4_window16_256.pth',

'SWINV2': {
    'EMBED_DIM': 96,             
    'DEPTHS': [2, 2, 18, 2],     
    'NUM_HEADS': [3, 6, 12, 24],   
    'WINDOW_SIZE': 16,
    'PRETRAINED_WINDOW_SIZES': [16, 16, 16, 8], 
}

```

---

## 🚀 Run the training script for your desired dataset:

**Train on Potsdam:**

```bash
python train.py --dataset potsdam

```

**Train on Vaihingen:**

```bash
python train.py --dataset vaihingen

```
---

## 📝 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{11482238,
  author={Li, Zicong and Li, Xiaotong and Zhu, Hao and Li, Weibin and Hou, Biao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FAD-Net: Frequency-Domain Amplitude-Phase Decoupling Network for Optical-Elevation Remote Sensing Segmentation}, 
  year={2026},
  volume={64},
  number={},
  pages={1-13},
  keywords={Feeds;Apertures;Antennas;Filtering;Filters;Circuits and systems;Gabor filters;Circuits;LoRa;High frequency;Feature fusion;frequency-domain analysis;multimodal remote sensing;semantic segmentation},
  doi={10.1109/TGRS.2026.3684237}}

