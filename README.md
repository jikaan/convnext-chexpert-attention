# ConvNeXt-CheXpert: Multi-Label Thoracic Disease Classification

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-View_Weights-yellow)](https://huggingface.co/calender/GRADCAM-Convnext-Chexpert-Attention)

## Abstract

This repository contains a PyTorch implementation of a **ConvNeXt-Base** architecture augmented with **Convolutional Block Attention Modules (CBAM)**. The model is fine-tuned on the **CheXpert** dataset to detect 14 common thoracic pathologies.

The project focuses on improving model interpretability in medical imaging by integrating channel and spatial attention mechanisms, validated through **Grad-CAM** visualization.

**Performance:** 0.81 mean AUC (Validation Set)
**Model Size:** 300MB (FP32)

## Directory Structure

To maintain a clean workspace, the repository is organized as follows:

```text
.
├── src/                    # Main source code
│   ├── gradcam_single.py   # GradCAM analysis script
│   └── training/           # Training scripts
│       └── train.py        # Main training loop
├── assets/                 # Model weights and visuals
│   ├── model_architecture/              # text file containing model architecture
│   └── analysis.png        # Sample visualizations
├── requirements.txt        # Dependencies
└── README.md
```
Installation
```
Bash

git clone [https://github.com/jikaan/convnext-chexpert-attention.git](https://github.com/jikaan/convnext-chexpert-attention.git)
cd convnext-chexpert-attention

pip install -r requirements.txt
```
Usage

1. Inference Example

Below is a snippet to run inference using the trained weights.
Python
```
import torch
from PIL import Image
from torchvision import transforms
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'assets/model/model.pth' # Update path as needed

model = timm.create_model('convnext_base', pretrained=False, num_classes=14)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.503] * 3, 
        std=[0.289] * 3
    )
])

image = Image.open('assets/sample_xray.jpg')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.sigmoid(model(input_tensor))

print(f"Pathology Probabilities: {probs[0].tolist()}")
```
2. Grad-CAM Visualization

To generate attention maps for a specific image, use the provided script in src:
Bash

python src/gradcam_single.py \
    --image assets/test_image.jpg \
    --model assets/model/model.pth \
    --output results.png

3. Training

To reproduce the training loop (Iteration 3):
Bash

python src/training/train.py \
    --data_dir /path/to/chexpert \
    --batch_size 4 \
    --epochs 3 \
    --lr 2e-5

Benchmarks

Evaluated on the CheXpert Validation Set.
Metric	Score	Configuration
Mean AUC	0.81	ConvNeXt-Base + CBAM
Input Size	384x384	Bicubic Interpolation
Optimizer	AdamW	Lookahead wrapper

Citation

Code snippet
```
@misc{convnext_cbam_2025,
  author = {Your Name},
  title = {ConvNeXt-CheXpert: Attention-Based Thoracic Classifier},
  year = {2025},
  publisher = {GitHub},
  url = {[https://github.com/jikaan/convnext-chexpert-attention](https://github.com/jikaan/convnext-chexpert-attention)}
}
```
License

This project is licensed under the Apache 2.0 License.
