"""
DenseNet121 model for skeleton keypoint sequence classification.
Adapted from SL-TSSI-DenseNet project for PyTorch.

Input shape: (batch_size, sequence_length, num_keypoints * 2)
- sequence_length: number of frames (variable)
- num_keypoints: number of skeleton joints (e.g., 45 for full body)
- 2: x, y coordinates

Architecture:
    Input → DenseNet121 (treating sequence as 2D image) → Global Avg Pool → Dropout → FC → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet1D(nn.Module):
    """
    1D DenseNet for temporal skeleton sequences.
    Uses conv1d layers to process temporal dimension.
    """
    def __init__(self, num_classes, input_channels=90, dropout=0.2):
        """
        Args:
            num_classes: Number of sign language classes
            input_channels: Number of input channels (num_keypoints * 2)
            dropout: Dropout rate before final classification layer
        """
        super(DenseNet1D, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # DenseNet blocks
        self.denseblock1 = self._make_dense_block(64, 6, growth_rate=32)
        self.transition1 = self._make_transition(64 + 6*32, (64 + 6*32)//2)
        
        self.denseblock2 = self._make_dense_block((64 + 6*32)//2, 12, growth_rate=32)
        self.transition2 = self._make_transition((64 + 6*32)//2 + 12*32, ((64 + 6*32)//2 + 12*32)//2)
        
        self.denseblock3 = self._make_dense_block(((64 + 6*32)//2 + 12*32)//2, 24, growth_rate=32)
        self.transition3 = self._make_transition(((64 + 6*32)//2 + 12*32)//2 + 24*32, 
                                                 (((64 + 6*32)//2 + 12*32)//2 + 24*32)//2)
        
        self.denseblock4 = self._make_dense_block((((64 + 6*32)//2 + 12*32)//2 + 24*32)//2, 16, growth_rate=32)
        
        # Final batch norm
        final_channels = (((64 + 6*32)//2 + 12*32)//2 + 24*32)//2 + 16*32
        self.bn_final = nn.BatchNorm1d(final_channels)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_channels, num_classes)
        
    def _make_dense_block(self, in_channels, num_layers, growth_rate=32):
        """Create a dense block"""
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)
    
    def _make_transition(self, in_channels, out_channels):
        """Create a transition layer"""
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, num_features)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Transpose to (batch_size, num_features, sequence_length) for conv1d
        x = x.transpose(1, 2)
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Dense blocks with transitions
        x = self.denseblock1(x)
        x = self.transition1(x)
        
        x = self.denseblock2(x)
        x = self.transition2(x)
        
        x = self.denseblock3(x)
        x = self.transition3(x)
        
        x = self.denseblock4(x)
        x = self.bn_final(x)
        x = self.relu(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove sequence dimension
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class DenseLayer(nn.Module):
    """Single dense layer in DenseNet"""
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm1d(4 * growth_rate)
        self.conv2 = nn.Conv1d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        new_features = self.conv1(self.relu(self.bn1(x)))
        new_features = self.conv2(self.relu(self.bn2(new_features)))
        return torch.cat([x, new_features], 1)


class DenseNetSkeleton(nn.Module):
    """
    DenseNet121 adapted for skeleton keypoint sequences.
    Treats the skeleton sequence as a pseudo-2D image.
    
    Similar to SL-TSSI-DenseNet approach.
    """
    def __init__(self, num_classes, num_keypoints=45, dropout=0.2, pretrained=False):
        """
        Args:
            num_classes: Number of sign language classes
            num_keypoints: Number of skeleton keypoints (default 45: body + 2 hands)
            dropout: Dropout rate
            pretrained: Whether to use ImageNet pretrained weights (usually False for skeleton data)
        """
        super(DenseNetSkeleton, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Load DenseNet121 backbone
        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
            densenet = densenet121(weights=weights)
        else:
            densenet = densenet121(weights=None)
        
        # Remove the original classifier
        self.features = densenet.features
        
        # Modify first conv layer to accept single channel input (we'll concatenate x,y as channels)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New: Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        original_conv = self.features.conv0
        self.features.conv0 = nn.Conv2d(
            2,  # 2 channels for x, y coordinates
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # If pretrained, initialize new conv layer with averaged weights
        if pretrained:
            with torch.no_grad():
                self.features.conv0.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1) / 2
                )
        
        # Global average pooling (already in DenseNet121: adaptive_avg_pool2d)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Get the number of features from DenseNet121
        num_features = densenet.classifier.in_features  # 1024 for DenseNet121
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_features, num_classes)
        
        print(f"✓ DenseNetSkeleton initialized: {num_keypoints} keypoints → {num_classes} classes")
        print(f"  Feature extractor: DenseNet121 (pretrained={pretrained})")
        print(f"  Dropout: {dropout}")
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, num_keypoints * 2)
               where num_keypoints * 2 represents [x1, y1, x2, y2, ..., xN, yN]
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len, features = x.shape
        
        # Reshape to (batch_size, num_keypoints, seq_len, 2)
        # Then permute to (batch_size, 2, seq_len, num_keypoints)
        # This treats the sequence as a 2D "image" with 2 channels (x, y)
        x = x.view(batch_size, seq_len, self.num_keypoints, 2)
        x = x.permute(0, 3, 1, 2)  # (batch, 2, seq_len, num_keypoints)
        
        # Pass through DenseNet features
        x = self.features(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def build_densenet_model(num_classes, 
                        num_keypoints=45, 
                        dropout=0.2, 
                        pretrained=False,
                        use_1d=False):
    """
    Build DenseNet model for skeleton-based sign language recognition.
    
    Args:
        num_classes: Number of sign language classes
        num_keypoints: Number of skeleton keypoints (default 45)
        dropout: Dropout rate (default 0.2)
        pretrained: Use ImageNet pretrained weights (default False)
        use_1d: Use 1D convolutions instead of 2D (default False)
    
    Returns:
        model: DenseNet model
    """
    if use_1d:
        model = DenseNet1D(
            num_classes=num_classes,
            input_channels=num_keypoints * 2,
            dropout=dropout
        )
        print(f"✓ Built DenseNet1D model with {num_keypoints} keypoints")
    else:
        model = DenseNetSkeleton(
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            dropout=dropout,
            pretrained=pretrained
        )
        print(f"✓ Built DenseNet2D model with {num_keypoints} keypoints")
    
    return model


if __name__ == "__main__":
    # Test the models
    print("Testing DenseNet models...\n")
    
    # Test DenseNetSkeleton (2D)
    model_2d = build_densenet_model(num_classes=100, num_keypoints=45, use_1d=False)
    test_input = torch.randn(4, 50, 90)  # (batch=4, seq_len=50, features=45*2)
    output = model_2d(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}\n")
    
    # Test DenseNet1D
    model_1d = build_densenet_model(num_classes=100, num_keypoints=45, use_1d=True)
    output = model_1d(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}\n")
    
    print("✅ All tests passed!")
