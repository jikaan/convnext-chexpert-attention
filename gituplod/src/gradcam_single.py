"""
Single Image Grad-CAM Analysis for ConvNeXt CheXpert Model

This script performs Grad-CAM and Layer-CAM analysis on a single chest X-ray image
to visualize what the model is focusing on for its predictions.

Usage:
    python gradcam_single.py --image path/to/xray.jpg --model path/to/model_cleaned.pth --output results.png
"""

import os
import torch
import timm
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image

# --- Correct imports from the 'pytorch-grad-cam' library ---
try:
    from pytorch_grad_cam import GradCAM, LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError("Please run 'pip install grad-cam scikit-learn' before executing this script.")


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class ConvNeXtWithMetadata(nn.Module):
    """ConvNeXt model with CBAM attention and metadata fusion"""
    def __init__(self, num_classes, metadata_input_dim, model_name="convnext_base", pretrained=False):
        super().__init__()
        self.convnext_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, features_only=True)
        self.cbam = CBAM(self.convnext_backbone.feature_info.channels()[-1], reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.num_image_features = self.convnext_backbone.feature_info.channels()[-1]
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2)
        )
        self.num_metadata_features = 32
        self.classifier = nn.Linear(self.num_image_features + self.num_metadata_features, num_classes)

    def forward(self, pixel_values, metadata):
        feats_list = self.convnext_backbone(pixel_values)
        feats = feats_list[-1]
        feats = self.cbam(feats)
        feats = self.global_pool(feats)
        feats = feats.view(feats.size(0), -1)
        metadata = metadata.float().to(feats.device)
        metadata_features = self.metadata_fc(metadata)
        combined_features = torch.cat((feats, metadata_features), dim=1)
        logits = self.classifier(combined_features)
        return logits


class GradCAMModelWrapper(nn.Module):
    """Wrapper for Grad-CAM that handles metadata"""
    def __init__(self, model, metadata_tensor):
        super().__init__()
        self.model = model
        self.register_buffer('metadata_tensor', metadata_tensor.clone())

    def forward(self, x):
        metadata = self.metadata_tensor.to(x.device)
        if metadata.size(0) != x.size(0):
            metadata = metadata[:1].expand(x.size(0), -1)
        return self.model(x, metadata)


def reshape_transform_convnext(tensor):
    """Transform for ConvNeXt features"""
    if len(tensor.shape) == 4:
        result = tensor.flatten(2).permute(0, 2, 1)
        return result
    return tensor


def analyze_single_image(image_path, model_path, confidence_threshold=0.5, output_path=None):
    """
    Analyze a single chest X-ray image with Grad-CAM and Layer-CAM

    Args:
        image_path: Path to the chest X-ray image
        model_path: Path to the trained model checkpoint
        confidence_threshold: Minimum confidence to analyze
        output_path: Optional path to save the visualization

    Returns:
        dict: Analysis results with predictions and visualizations
    """

    # Configuration
    IMAGE_SIZE = 384
    NUM_CLASSES = 14
    DATA_MEAN = [0.5029414296150208] * 3
    DATA_STD = [0.2892409563064575] * 3

    # Label columns
    LABEL_COLUMNS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = ConvNeXtWithMetadata(num_classes=NUM_CLASSES, metadata_input_dim=8).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle DataParallel wrapper
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)
    ])

    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Default metadata (you can customize this)
    metadata_tensor = torch.tensor([[
        0.0,  # sex (0 = male, 1 = female)
        50.0,  # age (scaled by 100)
        0.0,  # age missing indicator
        1.0,  # frontal/lateral (1 = frontal, 0 = lateral)
        0.0,  # AP (0 = PA, 1 = AP)
        1.0,  # PA (1 = PA, 0 = AP)
        0.0,  # AP/PA no label
        0.0   # AP/PA unknown
    ]], dtype=torch.float).to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(input_tensor, metadata_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Find top predictions above threshold
    top_indices = np.argsort(probabilities)[::-1]
    significant_findings = []

    for idx in top_indices:
        if probabilities[idx] > confidence_threshold:
            significant_findings.append({
                'class_idx': idx,
                'class_name': LABEL_COLUMNS[idx],
                'confidence': probabilities[idx]
            })

    if not significant_findings:
        print("No pathologies found above the confidence threshold.")
        return {'predictions': [], 'message': 'No significant findings'}

    print(f"Found {len(significant_findings)} significant findings above {confidence_threshold:.1%} threshold")

    # Find target layer for Grad-CAM
    target_layer = None
    for module in reversed(list(model.convnext_backbone.modules())):
        if isinstance(module, nn.Conv2d):
            target_layer = module
            break

    if target_layer is None:
        raise RuntimeError("Could not find a suitable convolution layer for Grad-CAM")

    # Analyze each significant finding
    results = {
        'image_path': image_path,
        'predictions': significant_findings,
        'visualizations': {}
    }

    cam_model = GradCAMModelWrapper(model, metadata_tensor)
    cam_model.eval()

    # Create figure for all visualizations
    num_findings = len(significant_findings)
    fig, axes = plt.subplots(num_findings, 5, figsize=(28, 7 * num_findings))
    if num_findings == 1:
        axes = axes.reshape(1, -1)

    for i, finding in enumerate(significant_findings):
        class_idx = finding['class_idx']
        class_name = finding['class_name']
        confidence = finding['confidence']

        print(f"\n--- Analyzing '{class_name}' (Confidence: {confidence:.1%}) ---")

        # Generate Grad-CAM and Layer-CAM
        targets = [ClassifierOutputTarget(class_idx)]

        with GradCAM(model=cam_model, target_layers=[target_layer]) as cam:
            grayscale_gradcam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        with LayerCAM(model=cam_model, target_layers=[target_layer], reshape_transform=reshape_transform_convnext) as cam:
            grayscale_layercam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Convert to RGB for visualization
        rgb_img = np.array(image_pil, dtype=np.float32) / 255.0

        # Resize heatmaps to match image size
        grayscale_gradcam_resized = cv2.resize(grayscale_gradcam, (rgb_img.shape[1], rgb_img.shape[0]))
        grayscale_layercam_resized = cv2.resize(grayscale_layercam, (rgb_img.shape[1], rgb_img.shape[0]))

        # Create overlay visualizations
        gradcam_visualization = show_cam_on_image(rgb_img, grayscale_gradcam_resized, use_rgb=True, image_weight=0.5, colormap=cv2.COLORMAP_JET)
        layercam_visualization = show_cam_on_image(rgb_img, grayscale_layercam_resized, use_rgb=True, image_weight=0.5, colormap=cv2.COLORMAP_JET)

        # Plot results
        row_axes = axes[i]

        # Original X-ray with disease name
        row_axes[0].imshow(image_pil, cmap='gray')
        row_axes[0].set_title(f"Original X-ray\n{class_name}", fontsize=12, pad=20)
        row_axes[0].axis('off')

        # Grad-CAM heatmap
        row_axes[1].imshow(grayscale_gradcam_resized, cmap='jet')
        row_axes[1].set_title(f"Grad-CAM Heatmap\n{confidence:.1%}", fontsize=12, pad=20)
        row_axes[1].axis('off')

        # Grad-CAM overlay
        row_axes[2].imshow(gradcam_visualization)
        row_axes[2].set_title(f"Grad-CAM Overlay\n{class_name} ({confidence:.1%})", fontsize=12, pad=20)
        row_axes[2].axis('off')

        # Layer-CAM heatmap
        row_axes[3].imshow(grayscale_layercam_resized, cmap='jet')
        row_axes[3].set_title(f"Layer-CAM Heatmap\n{confidence:.1%}", fontsize=12, pad=20)
        row_axes[3].axis('off')

        # Layer-CAM overlay
        row_axes[4].imshow(layercam_visualization)
        row_axes[4].set_title(f"Layer-CAM Overlay\n{class_name} ({confidence:.1%})", fontsize=12, pad=20)
        row_axes[4].axis('off')

        # Set overall title for this finding
        if num_findings > 1:
            fig.suptitle(f"Grad-CAM Analysis Results", fontsize=16, y=0.98)

        # Add disease name as a subtitle for this row (for multiple findings)
        if num_findings > 1:
            # Create a text annotation above this row
            fig.text(0.5, 0.98 - (i * 0.15), f"Disease: {class_name} (Confidence: {confidence:.1%})",
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # Store visualization data
        results['visualizations'][class_name] = {
            'gradcam_heatmap': grayscale_gradcam_resized,
            'layercam_heatmap': grayscale_layercam_resized,
            'gradcam_overlay': gradcam_visualization,
            'layercam_overlay': layercam_visualization
        }

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Single Image Grad-CAM Analysis")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", help="Output path for visualization (optional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")

    args = parser.parse_args()

    results = analyze_single_image(
        image_path=args.image,
        model_path=args.model,
        confidence_threshold=args.threshold,
        output_path=args.output
    )

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Findings above {args.threshold:.1%} threshold: {len(results['predictions'])}")

    for finding in results['predictions']:
        print(f"  - {finding['class_name']}: {finding['confidence']:.1%}")


if __name__ == "__main__":
    main()
