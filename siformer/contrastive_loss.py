"""
Supervised Contrastive Loss and Center Loss for Sign Language Recognition

This module implements feature-level regularization strategies:

1. Supervised Contrastive Loss: Pulls same-class samples together, pushes different-class samples apart
2. Center Loss: Minimizes intra-class variation by pulling features toward class centers
3. Hybrid Loss: Combines both strategies

Paper: "Joint Contrastive and Center Loss Regularization for Robust Sign Language Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    Based on: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    Adapted for sign language recognition.
    
    In a batch, samples from the same class should have similar features,
    while samples from different classes should have dissimilar features.
    
    Args:
        temperature: Scaling factor for similarity (default: 0.07)
        contrast_mode: 'all' or 'one' - how to construct positive pairs
        base_temperature: Base temperature for normalization (default: 0.07)
    """
    
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
                      Should be L2-normalized
            labels: Ground truth labels (batch_size,)
        
        Returns:
            loss: Scalar contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        # (batch_size, batch_size)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to exclude self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        # Handle case where a class has only one sample
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class CenterLoss(nn.Module):
    """
    Center Loss for feature learning.
    
    Based on: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    (Wen et al., ECCV 2016)
    
    Learns a center (mean) for each class and penalizes the distance between
    features and their corresponding class centers.
    
    Args:
        num_classes: Number of classes
        feature_dim: Dimension of feature representations
        lambda_center: Weight for center loss (default: 0.003)
        use_cuda: Whether to use CUDA
    """
    
    def __init__(self, num_classes: int, feature_dim: int, 
                 lambda_center: float = 0.003, use_cuda: bool = True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_center = lambda_center
        self.use_cuda = use_cuda
        
        # Initialize centers (num_classes, feature_dim)
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
        if use_cuda and torch.cuda.is_available():
            self.centers = self.centers.cuda()
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            loss: Scalar center loss
        """
        batch_size = features.size(0)
        
        # Get centers for this batch
        # (batch_size, feature_dim)
        centers_batch = self.centers.index_select(0, labels.long())
        
        # Compute distances to centers
        # ||f - c||^2
        loss = (features - centers_batch).pow(2).sum() / batch_size
        
        return loss * self.lambda_center
    
    def get_centers(self) -> torch.Tensor:
        """Return current class centers."""
        return self.centers.data
    
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor, 
                      alpha: float = 0.5):
        """
        Update centers with moving average.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)
            alpha: Update rate (0 = no update, 1 = full update)
        """
        with torch.no_grad():
            for label in labels.unique():
                # Get features for this class
                mask = (labels == label)
                class_features = features[mask]
                
                # Compute mean
                class_mean = class_features.mean(dim=0)
                
                # Update center with moving average
                self.centers[label] = (1 - alpha) * self.centers[label] + alpha * class_mean


class PrototypeLoss(nn.Module):
    """
    Prototype-based regularization loss.
    
    Similar to Center Loss but uses prototypes computed from current batch
    instead of maintained centers.
    
    Args:
        temperature: Temperature for prototype assignment (default: 0.1)
    """
    
    def __init__(self, temperature: float = 0.1):
        super(PrototypeLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype loss.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            loss: Scalar prototype loss
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute prototypes for each class in batch
        unique_labels = labels.unique()
        loss = 0.0
        num_samples = 0
        
        for label in unique_labels:
            # Get features for this class
            mask = (labels == label)
            class_features = features[mask]
            
            # Compute prototype (mean)
            prototype = class_features.mean(dim=0, keepdim=True)
            prototype = F.normalize(prototype, p=2, dim=1)
            
            # Compute distance to prototype
            distances = 1 - torch.mm(class_features, prototype.t())
            loss += distances.sum()
            num_samples += class_features.size(0)
        
        return loss / max(num_samples, 1)


class HybridContrastiveCenterLoss(nn.Module):
    """
    Combines Supervised Contrastive Loss and Center Loss.
    
    Total loss = CE_loss + beta * SupCon_loss + gamma * Center_loss
    
    Args:
        num_classes: Number of classes
        feature_dim: Dimension of features
        temperature: Temperature for contrastive loss (default: 0.07)
        lambda_center: Weight for center loss (default: 0.003)
        beta: Weight for contrastive loss (default: 0.5)
        gamma: Weight for center loss (default: 0.1)
        use_cuda: Whether to use CUDA
    """
    
    def __init__(self, num_classes: int, feature_dim: int,
                 temperature: float = 0.07, lambda_center: float = 0.003,
                 beta: float = 0.5, gamma: float = 0.1,
                 use_cuda: bool = True):
        super(HybridContrastiveCenterLoss, self).__init__()
        
        self.supcon_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.center_loss = CenterLoss(num_classes, feature_dim, 
                                     lambda_center=lambda_center, use_cuda=use_cuda)
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            dict containing:
                - 'total': Total auxiliary loss
                - 'contrastive': Contrastive loss component
                - 'center': Center loss component
        """
        # Compute individual losses
        contrastive_loss = self.supcon_loss(features, labels)
        center_loss = self.center_loss(features, labels)
        
        # Weighted sum
        total_loss = self.beta * contrastive_loss + self.gamma * center_loss
        
        return {
            'total': total_loss,
            'contrastive': contrastive_loss,
            'center': center_loss
        }
    
    def get_centers(self) -> torch.Tensor:
        """Return current class centers."""
        return self.center_loss.get_centers()
    
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor, 
                      alpha: float = 0.5):
        """Update centers with moving average."""
        self.center_loss.update_centers(features, labels, alpha)


def compute_feature_statistics(features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about feature distribution for analysis.
    
    Args:
        features: Feature representations (batch_size, feature_dim)
        labels: Ground truth labels (batch_size,)
    
    Returns:
        dict containing:
            - 'intra_class_variance': Mean variance within classes
            - 'inter_class_distance': Mean distance between class centers
            - 'feature_norm': Mean L2 norm of features
    """
    features = features.detach().cpu()
    labels = labels.detach().cpu()
    
    # Normalize features
    features_norm = F.normalize(features, p=2, dim=1)
    
    # Compute intra-class variance
    unique_labels = labels.unique()
    intra_variances = []
    class_centers = []
    
    for label in unique_labels:
        mask = (labels == label)
        class_features = features_norm[mask]
        
        if class_features.size(0) > 1:
            center = class_features.mean(dim=0)
            variance = ((class_features - center) ** 2).sum(dim=1).mean().item()
            intra_variances.append(variance)
            class_centers.append(center)
    
    intra_class_variance = np.mean(intra_variances) if intra_variances else 0.0
    
    # Compute inter-class distance
    if len(class_centers) > 1:
        class_centers = torch.stack(class_centers)
        # Compute pairwise distances
        distances = torch.cdist(class_centers, class_centers, p=2)
        # Exclude diagonal (self-distance)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool)
        inter_class_distance = distances[mask].mean().item()
    else:
        inter_class_distance = 0.0
    
    # Compute mean feature norm
    feature_norm = features.norm(p=2, dim=1).mean().item()
    
    return {
        'intra_class_variance': intra_class_variance,
        'inter_class_distance': inter_class_distance,
        'feature_norm': feature_norm
    }


if __name__ == "__main__":
    # Test contrastive and center losses
    print("Testing Contrastive and Center Loss Module\n")
    print("=" * 70)
    
    # Create dummy data
    batch_size = 32
    feature_dim = 256
    num_classes = 100
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Make some samples from same class for testing
    labels[:8] = 0  # 8 samples from class 0
    labels[8:16] = 1  # 8 samples from class 1
    
    print(f"Input:")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Num unique classes in batch: {labels.unique().size(0)}")
    print()
    
    # Test 1: Supervised Contrastive Loss
    print("Test 1: Supervised Contrastive Loss")
    supcon = SupervisedContrastiveLoss(temperature=0.07)
    loss_con = supcon(features, labels)
    print(f"  - Contrastive Loss: {loss_con.item():.4f}")
    print()
    
    # Test 2: Center Loss
    print("Test 2: Center Loss")
    center_loss = CenterLoss(num_classes, feature_dim, lambda_center=0.003, use_cuda=False)
    loss_center = center_loss(features, labels)
    print(f"  - Center Loss: {loss_center.item():.6f}")
    print()
    
    # Test 3: Hybrid Loss
    print("Test 3: Hybrid Contrastive + Center Loss")
    hybrid = HybridContrastiveCenterLoss(
        num_classes, feature_dim,
        beta=0.5, gamma=0.1, use_cuda=False
    )
    losses = hybrid(features, labels)
    print(f"  - Total Loss: {losses['total'].item():.4f}")
    print(f"  - Contrastive: {losses['contrastive'].item():.4f}")
    print(f"  - Center: {losses['center'].item():.6f}")
    print()
    
    # Test 4: Feature Statistics
    print("Test 4: Feature Statistics")
    stats = compute_feature_statistics(features, labels)
    print(f"  - Intra-class variance: {stats['intra_class_variance']:.4f}")
    print(f"  - Inter-class distance: {stats['inter_class_distance']:.4f}")
    print(f"  - Feature norm: {stats['feature_norm']:.4f}")
    print()
    
    # Test 5: Gradient flow
    print("Test 5: Gradient Flow")
    features.requires_grad = True
    losses = hybrid(features, labels)
    losses['total'].backward()
    print(f"  - Feature gradient norm: {features.grad.norm().item():.4f}")
    print(f"  - Gradient flow: ✓")
    print()
    
    print("=" * 70)
    print("All tests passed! ✓")
