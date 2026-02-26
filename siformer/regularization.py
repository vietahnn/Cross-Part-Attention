"""
Advanced Regularization Techniques for Overfitting Prevention
============================================================
This module implements multiple regularization strategies:
1. Stochastic Depth (DropPath)
2. Feature Dropout
3. Spatial Dropout
4. Adaptive Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DropPath(nn.Module):
    """
    Stochastic Depth / Drop Path
    Randomly drops residual connections during training.
    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class FeatureDropout(nn.Module):
    """
    Feature-level dropout for multi-part models
    Drops entire features (body parts) during training
    """
    def __init__(self, drop_prob=0.1):
        super(FeatureDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # x shape: [seq_len, batch, features]
        mask = torch.rand(1, x.shape[1], x.shape[2], device=x.device) > self.drop_prob
        return x * mask.float() / (1 - self.drop_prob)


class SpatialDropout1D(nn.Module):
    """
    Spatial Dropout for sequential data
    Drops entire feature maps across time dimension
    """
    def __init__(self, drop_prob=0.1):
        super(SpatialDropout1D, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # x shape: [seq_len, batch, features]
        # Drop entire channels across the sequence
        mask = torch.rand(1, x.shape[1], x.shape[2], device=x.device) > self.drop_prob
        return x * mask.float() / (1 - self.drop_prob)


class AdaptiveDropout(nn.Module):
    """
    Adaptive dropout that increases during training
    Starts with lower dropout and increases over epochs
    """
    def __init__(self, initial_drop=0.1, max_drop=0.3):
        super(AdaptiveDropout, self).__init__()
        self.initial_drop = initial_drop
        self.max_drop = max_drop
        self.current_drop = initial_drop
        
    def set_drop_rate(self, progress):
        """
        progress: float from 0.0 to 1.0 indicating training progress
        """
        self.current_drop = self.initial_drop + (self.max_drop - self.initial_drop) * progress
        
    def forward(self, x):
        if not self.training:
            return x
        return F.dropout(x, p=self.current_drop, training=True)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Prevents overconfident predictions
    Reference: "Rethinking the Inception Architecture" (Szegedy et al., 2016)
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size, num_classes] - logits
            target: [batch_size] - class indices
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.shape[-1] - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class MixupAugmentation:
    """
    Mixup Data Augmentation
    Mixes two samples and their labels
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Args:
            x: input data [batch, ...]
            y: labels [batch]
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class CutMixAugmentation:
    """
    CutMix Data Augmentation for sequential data
    Cuts and pastes patches between samples
    Reference: "CutMix: Regularization Strategy" (Yun et al., 2019)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Args:
            x: input data [seq_len, batch, features]
            y: labels [batch]
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(1)
        index = torch.randperm(batch_size).to(x.device)

        # Cut a portion of the sequence
        seq_len = x.size(0)
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1) if cut_len < seq_len else 0

        x_mixed = x.clone()
        x_mixed[cut_start:cut_start+cut_len, :, :] = x[cut_start:cut_start+cut_len, index, :]
        
        y_a, y_b = y, y[index]
        return x_mixed, y_a, y_b, lam


class TemporalDropout(nn.Module):
    """
    Temporal Dropout - randomly drops time steps in sequences
    Useful for sign language recognition to simulate occlusions
    """
    def __init__(self, drop_prob=0.1):
        super(TemporalDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # x shape: [seq_len, batch, features]
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_prob
        return x * mask.float()


class MultiPartDropout(nn.Module):
    """
    Dropout strategy specifically for multi-part sign language models
    Randomly drops entire body parts (left hand, right hand, or body)
    """
    def __init__(self, drop_prob=0.1, part_dims=[21, 21, 42]):
        """
        Args:
            drop_prob: probability of dropping each part
            part_dims: dimensions of [left_hand, right_hand, body]
        """
        super(MultiPartDropout, self).__init__()
        self.drop_prob = drop_prob
        self.part_dims = part_dims
        self.cumsum = np.cumsum([0] + part_dims)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        # Randomly decide which parts to drop
        for i in range(len(self.part_dims)):
            if torch.rand(1).item() < self.drop_prob:
                start_idx = self.cumsum[i]
                end_idx = self.cumsum[i + 1]
                x[:, :, start_idx:end_idx] = 0
        
        return x


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixup loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
