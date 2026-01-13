# FAD-Net: Frequency-Domain Amplitude-Phase Decoupling Network for Optical-Elevation Remote Sensing Segmentation

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
