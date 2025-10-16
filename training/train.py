"""
ConvNeXt-based Multi-label Chest X-ray Classification Model

This module implements a ConvNeXt-based architecture for multi-label classification
of chest X-ray images using the CheXpert dataset. The model incorporates:
- ConvNeXt backbone pretrained on ImageNet
- CBAM attention mechanism
- Metadata fusion for patient demographics

Architecture Overview:
- Backbone: ConvNeXt-Base (pretrained on ImageNet-22K)
- Attention: Convolutional Block Attention Module (CBAM)
- Fusion: Concatenation of image features and metadata features
- Output: 14-class multi-label classification

Usage:
    from convnext_chexpert import ConvNeXtWithMetadata

    # Initialize model
    model = ConvNeXtWithMetadata(
        num_classes=14,
        metadata_input_dim=8,
        model_name="convnext_base",
        pretrained=True
    )

    # Load trained weights
    state_dict = torch.load('model_weights.pth')
    model.load_state_dict(state_dict['model_state_dict'])
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)

    Implements both channel and spatial attention mechanisms to improve
    feature representation quality for medical image analysis.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
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
    """
    ConvNeXt model with CBAM attention and metadata fusion for chest X-ray classification.

    This model combines:
    1. ConvNeXt backbone for image feature extraction
    2. CBAM attention module for enhanced feature representation
    3. Metadata processing for patient demographics
    4. Multi-label classification head

    Args:
        num_classes: Number of output classes (14 for CheXpert)
        metadata_input_dim: Dimension of metadata input (8 for CheXpert)
        model_name: Backbone model name (default: convnext_base)
        pretrained: Whether to use pretrained ImageNet weights
    """
    def __init__(self, num_classes, metadata_input_dim, model_name="convnext_base", pretrained=True):
        super().__init__()

        # Load ConvNeXt backbone
        try:
            import timm
            self.convnext_backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, features_only=True
            )
            # Get number of features from last layer
            backbone_channels = self.convnext_backbone.feature_info.channels()[-1]
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        # CBAM attention module
        self.cbam = CBAM(channels=backbone_channels, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Metadata processing network
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classification head
        self.classifier = nn.Linear(backbone_channels + 32, num_classes)

    def forward(self, pixel_values, metadata):
        """
        Forward pass through the network.

        Args:
            pixel_values: Input chest X-ray images [B, 3, H, W]
            metadata: Patient metadata [B, metadata_input_dim]

        Returns:
            logits: Raw classification logits [B, num_classes]
        """
        # Extract features from ConvNeXt backbone
        feats = self.convnext_backbone(pixel_values)[-1]  # Get last feature map

        # Apply CBAM attention
        feats = self.cbam(feats)

        # Global average pooling
        feats = self.global_pool(feats)
        feats = feats.view(feats.size(0), -1)  # Flatten

        # Process metadata
        metadata = metadata.float().to(feats.device)
        metadata_features = self.metadata_fc(metadata)

        # Concatenate image and metadata features
        combined_features = torch.cat((feats, metadata_features), dim=1)

        # Final classification
        logits = self.classifier(combined_features)
        return logits


def create_model(num_classes=14, metadata_input_dim=8, pretrained=True):
    """
    Create a ConvNeXt model with CBAM and metadata fusion.

    Args:
        num_classes: Number of output classes
        metadata_input_dim: Dimension of metadata input
        pretrained: Whether to use pretrained weights

    Returns:
        model: ConvNeXtWithMetadata model instance
    """
    model = ConvNeXtWithMetadata(
        num_classes=num_classes,
        metadata_input_dim=metadata_input_dim,
        model_name="convnext_base",
        pretrained=pretrained
    )
    return model


def load_model_weights(model, checkpoint_path, device='cpu'):
    """
    Load model weights from checkpoint file.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to map weights to

    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle DataParallel wrapper if present
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights loaded successfully.")

    return model


def predict(model, image, metadata, device='cpu'):
    """
    Make predictions on a single image with metadata.

    Args:
        model: Trained model
        image: Input image tensor [3, H, W]
        metadata: Patient metadata tensor [8]
        device: Device to run inference on

    Returns:
        probabilities: Predicted probabilities for each class [14]
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        metadata = metadata.unsqueeze(0).to(device)  # Add batch dimension

        logits = model(image, metadata)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    return probabilities


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model(
        num_classes=14,
        metadata_input_dim=8,
        pretrained=True
    )

    print("ConvNeXt CheXpert model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")