# ConvNeXt-CheXpert-Attention 🏥

[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Hub-yellow)](https://huggingface.co/calender/Convnext-Chexpert-Attention)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Multi-label chest X-ray classifier: **ConvNeXt-Base + CBAM attention** trained on CheXpert to detect 14 thoracic pathologies.

**Model AUC: 0.81** | **Iteration 6** | **GradCAM Enabled** | **300MB**

🤗 [Get Model Weights on HuggingFace](https://huggingface.co/calender/Convnext-Chexpert-Attention)

---

## ⚡ Quick Demo

```python
import torch
from PIL import Image
from torchvision import transforms
import timm

# Load model
model = timm.create_model('convnext_base', pretrained=False, num_classes=14)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Predict
image = Image.open('chest_xray.jpg')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.503]*3, std=[0.289]*3)
])

with torch.no_grad():
    probs = torch.sigmoid(model(transform(image).unsqueeze(0)))
    
# Results: 14 pathology probabilities
```

See `examples/inference_example.py` for full code.

---

## 📖 What This Does

Analyzes chest X-rays and predicts **14 pathologies**:

Edema • Cardiomegaly • Pleural Effusion • Atelectasis • Consolidation • Pneumonia • Fracture • Lung Opacity • Pneumothorax • Lung Lesion • Cardiomediastinum • Pleural Other • Support Devices • No Finding

**Includes GradCAM visualization** to see which regions the model focuses on for each prediction.

---

## 🚀 Getting Started

### 1. Setup

```bash
git clone https://github.com/jikaan/convnext-chexpert-attention.git
cd convnext-chexpert-attention

pip install -r requirements.txt

# Download model from HuggingFace
# https://huggingface.co/calender/Convnext-Chexpert-Attention
```

### 2. Inference

```bash
# Basic usage (requires model file)
python gituplod/src/gradcam_single.py --image path/to/xray.jpg --model hfupld/model/model.pth
```

### 3. GradCAM Visualization

```bash
# Single image analysis with multiple findings
python gituplod/src/gradcam_single.py --image path/to/xray.jpg --model hfupld/model/model.pth --output results.png
```

See `hfupld/` folder for sample visualizations (1.png, 2.png, 3.png, analysis.png).

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Validation AUC | 0.81 |
| Architecture | ConvNeXt-Base + CBAM |
| Dataset | CheXpert (224K images) |
| Input Size | 384×384 |
| Model Size | 351MB |

**Examples from Test Set:**

| Pathology | Confidence | GradCAM |
|-----------|-----------|---------|
| Edema | 63.7% | ![](../hfupld/analysis.png) |
| Multiple Findings | 65.2% | ![](../hfupld/1.png) |
| Cardiomegaly | 67.2% | ![](../hfupld/2.png) |
| Pneumothorax | 63.1% | ![](../hfupld/3.png) |

---

## 📁 Structure

```
chexpert-convnext-classifier/
├── gituplod/                          # Repository files
│   ├── src/
│   │   └── gradcam_single.py          # Single image GradCAM analysis
│   ├── training/
│   │   └── train.py                   # Training script (iteration 3)
│   ├── examples/
│   │   └── test_images/               # Sample visualization outputs
│   ├── LICENSE                        # Apache 2.0 License
│   ├── README.md                      # This file
│   └── requirements.txt               # Python dependencies
├── hfupld/                           # Model and demo files
│   ├── model/
│   │   ├── model.pth (351MB)          # Model weights
│   │   └── model_config.json          # Model configuration
│   ├── 1.png, 2.png, 3.png            # Single pathology GradCAM examples
│   ├── analysis.png                   # Multi-disease visualization
│   └── README.md                      # Model card
└── requirements.txt                   # Project dependencies
```

---

## 🏋️ Training

Reproduce results with iteration 3 training script:

```bash
python gituplod/training/train.py \
    --data_dir path/to/chexpert \
    --batch_size 4 \
    --epochs 3 \
    --lr 2e-5
```

**Training Config:**
- Optimizer: AdamW + Lookahead
- Loss: Focal Loss + uncertainty masking
- Augmentation: Crops, flips, rotation, color jitter
- Dataset: CheXpert (88% train, 2% val, 10% test)

See `gituplod/training/train.py` for full implementation.

---

## 💡 Key Features

✅ **CBAM Attention** - Better pathology localization  
✅ **GradCAM Support** - Visual explanations  
✅ **Multi-label** - Detects multiple pathologies simultaneously  
✅ **Uncertainty Handling** - Trained with uncertain labels  
✅ **Clean Code** - Iteration 3 training script included  

---

## ⚠️ Important

**This is research code. Medical Disclaimer:**
- NOT for clinical diagnosis
- NOT FDA-approved
- Requires expert radiologist review
- See [LICENSE](LICENSE) for full terms

---

## 📝 Citation

If you use this in research:

```bibtex
@software{convnext_chexpert_attention_2025,
  author = {Time},
  title = {ConvNeXt-Base CheXpert Classifier with CBAM Attention},
  year = {2025},
  url = {https://github.com/jikaan/convnext-chexpert-attention}
}

@article{irvin2019chexpert,
  title={CheXpert: A large chest radiograph dataset with uncertainty labels},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and others},
  year={2019}
}
```

---

## 🔗 Links

- **Model (HuggingFace):** https://huggingface.co/calender/Convnext-Chexpert-Attention
- **CheXpert Dataset:** https://stanfordmlgroup.github.io/competitions/chexpert/
- **Paper:** https://arxiv.org/abs/1901.07031

---

## 📧 Contact

Questions? Open an issue on GitHub.

**Created by Time | October 2025**