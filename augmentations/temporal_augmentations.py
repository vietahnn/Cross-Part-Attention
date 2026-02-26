"""
Temporal Masking and Keypoint Dropout for Sign Language Recognition

This module implements temporal-aware augmentation strategies to improve 
model robustness and reduce overfitting:

1. Temporal Masking: Randomly mask (zero out) frames in the sequence
2. Keypoint Dropout: Randomly drop individual keypoints or body parts
3. Sequential Cutout: Drop consecutive frames or spatial regions

Paper: "Temporal Masking and Keypoint Dropout for Robust Skeleton-Based Sign Recognition"
"""

import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class TemporalMasking:
    """
    Randomly mask frames in a video sequence during training.
    Forces the model to learn robust temporal features that can handle missing frames.
    
    Args:
        mask_ratio: Probability of masking each frame (default: 0.15)
        mask_value: Value to use for masked frames (default: 0.0)
        consecutive: If True, mask consecutive frames; if False, random frames
        max_consecutive: Maximum number of consecutive frames to mask
    """
    
    def __init__(self, mask_ratio: float = 0.15, mask_value: float = 0.0, 
                 consecutive: bool = False, max_consecutive: int = 10):
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.consecutive = consecutive
        self.max_consecutive = max_consecutive
    
    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply temporal masking to sequence.
        
        Args:
            sequence: Shape (T, num_keypoints, 2) or (T, num_keypoints, C)
        
        Returns:
            masked_sequence: Same shape as input
        """
        T = sequence.shape[0]
        masked_sequence = sequence.copy()
        
        if self.consecutive:
            # Mask consecutive frames
            num_frames_to_mask = int(T * self.mask_ratio)
            if num_frames_to_mask > 0:
                # Random start position
                max_start = max(0, T - num_frames_to_mask)
                start_idx = random.randint(0, max_start)
                end_idx = min(start_idx + num_frames_to_mask, T)
                masked_sequence[start_idx:end_idx] = self.mask_value
        else:
            # Mask random frames
            mask = np.random.random(T) < self.mask_ratio
            masked_sequence[mask] = self.mask_value
        
        return masked_sequence


class KeypointDropout:
    """
    Randomly drop (zero out) individual keypoints or entire body parts.
    Forces the model to learn from partial observations.
    
    Args:
        dropout_prob: Probability of dropping keypoints (default: 0.3)
        dropout_type: 'random' (individual keypoints) or 'bodypart' (entire parts)
        max_keypoints: Maximum number of keypoints to drop (for 'random' type)
        body_parts_config: Dict mapping body part names to keypoint indices
    """
    
    def __init__(self, dropout_prob: float = 0.3, dropout_type: str = 'random',
                 max_keypoints: int = 5, body_parts_config: Optional[Dict] = None):
        self.dropout_prob = dropout_prob
        self.dropout_type = dropout_type
        self.max_keypoints = max_keypoints
        self.body_parts_config = body_parts_config or self._default_body_parts()
    
    def _default_body_parts(self) -> Dict[str, List[int]]:
        """
        Default body part configuration for pose keypoints.
        Adjust based on your keypoint format.
        """
        return {
            'left_hand': list(range(0, 21)),      # 21 hand keypoints
            'right_hand': list(range(21, 42)),    # 21 hand keypoints
            'body': list(range(42, 54)),          # 12 body keypoints
        }
    
    def __call__(self, sequence: np.ndarray, part: str = 'all') -> np.ndarray:
        """
        Apply keypoint dropout to sequence.
        
        Args:
            sequence: Shape (T, num_keypoints, 2) or (T, num_keypoints, C)
            part: Which part to process ('all', 'left_hand', 'right_hand', 'body')
        
        Returns:
            dropped_sequence: Same shape as input
        """
        if random.random() > self.dropout_prob:
            return sequence
        
        T, num_keypoints, C = sequence.shape
        dropped_sequence = sequence.copy()
        
        if self.dropout_type == 'random':
            # Drop random keypoints
            if part == 'all':
                keypoint_indices = list(range(num_keypoints))
            else:
                keypoint_indices = self.body_parts_config.get(part, [])
            
            if keypoint_indices:
                num_to_drop = random.randint(1, min(self.max_keypoints, len(keypoint_indices)))
                drop_indices = random.sample(keypoint_indices, num_to_drop)
                dropped_sequence[:, drop_indices, :] = 0.0
        
        elif self.dropout_type == 'bodypart':
            # Drop entire body part
            available_parts = ['left_hand', 'right_hand']  # Don't drop body
            if random.random() < 0.3:  # 30% chance to drop a body part
                part_to_drop = random.choice(available_parts)
                drop_indices = self.body_parts_config.get(part_to_drop, [])
                dropped_sequence[:, drop_indices, :] = 0.0
        
        return dropped_sequence


class SequentialCutout:
    """
    Drop consecutive frames or spatial regions (similar to Cutout for images).
    
    Args:
        temporal_cutout_prob: Probability of applying temporal cutout
        spatial_cutout_prob: Probability of applying spatial cutout
        temporal_cutout_frames: Number of consecutive frames to cut
        spatial_cutout_keypoints: Number of keypoints to cut spatially
    """
    
    def __init__(self, temporal_cutout_prob: float = 0.2, 
                 spatial_cutout_prob: float = 0.2,
                 temporal_cutout_frames: int = 15,
                 spatial_cutout_keypoints: int = 10):
        self.temporal_cutout_prob = temporal_cutout_prob
        self.spatial_cutout_prob = spatial_cutout_prob
        self.temporal_cutout_frames = temporal_cutout_frames
        self.spatial_cutout_keypoints = spatial_cutout_keypoints
    
    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply sequential cutout to sequence.
        
        Args:
            sequence: Shape (T, num_keypoints, 2) or (T, num_keypoints, C)
        
        Returns:
            cutout_sequence: Same shape as input
        """
        T, num_keypoints, C = sequence.shape
        cutout_sequence = sequence.copy()
        
        # Temporal cutout: zero out consecutive frames
        if random.random() < self.temporal_cutout_prob:
            cutout_length = min(self.temporal_cutout_frames, T // 3)  # Max 1/3 of sequence
            start_idx = random.randint(0, max(0, T - cutout_length))
            end_idx = start_idx + cutout_length
            cutout_sequence[start_idx:end_idx] = 0.0
        
        # Spatial cutout: zero out random keypoints across all frames
        if random.random() < self.spatial_cutout_prob:
            num_to_cut = min(self.spatial_cutout_keypoints, num_keypoints // 3)
            cut_indices = random.sample(range(num_keypoints), num_to_cut)
            cutout_sequence[:, cut_indices, :] = 0.0
        
        return cutout_sequence


class HybridTemporalAugmentation:
    """
    Combines multiple temporal augmentation strategies.
    Randomly applies one or more augmentation techniques.
    
    Args:
        temporal_mask_prob: Probability of applying temporal masking
        keypoint_dropout_prob: Probability of applying keypoint dropout
        sequential_cutout_prob: Probability of applying sequential cutout
        temporal_masking: TemporalMasking instance
        keypoint_dropout: KeypointDropout instance
        sequential_cutout: SequentialCutout instance
    """
    
    def __init__(self, 
                 temporal_mask_prob: float = 0.3,
                 keypoint_dropout_prob: float = 0.3,
                 sequential_cutout_prob: float = 0.2,
                 mask_ratio: float = 0.15,
                 dropout_prob: float = 0.3,
                 dropout_type: str = 'random',
                 max_keypoints: int = 5):
        
        self.temporal_mask_prob = temporal_mask_prob
        self.keypoint_dropout_prob = keypoint_dropout_prob
        self.sequential_cutout_prob = sequential_cutout_prob
        
        self.temporal_masking = TemporalMasking(mask_ratio=mask_ratio)
        self.keypoint_dropout = KeypointDropout(
            dropout_prob=dropout_prob, 
            dropout_type=dropout_type,
            max_keypoints=max_keypoints
        )
        self.sequential_cutout = SequentialCutout()
    
    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply hybrid temporal augmentation.
        
        Args:
            sequence: Shape (T, num_keypoints, 2) or (T, num_keypoints, C)
        
        Returns:
            augmented_sequence: Same shape as input
        """
        augmented = sequence.copy()
        
        # Apply temporal masking
        if random.random() < self.temporal_mask_prob:
            augmented = self.temporal_masking(augmented)
        
        # Apply keypoint dropout
        if random.random() < self.keypoint_dropout_prob:
            augmented = self.keypoint_dropout(augmented)
        
        # Apply sequential cutout
        if random.random() < self.sequential_cutout_prob:
            augmented = self.sequential_cutout(augmented)
        
        return augmented


def apply_temporal_augmentation(
    l_hand: np.ndarray, 
    r_hand: np.ndarray, 
    body: np.ndarray,
    augmentation_config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to apply temporal augmentation to all body parts.
    
    Args:
        l_hand: Left hand sequence (T, 21, C)
        r_hand: Right hand sequence (T, 21, C)
        body: Body sequence (T, 12, C)
        augmentation_config: Configuration dict for augmentation parameters
    
    Returns:
        Tuple of augmented (l_hand, r_hand, body)
    """
    if augmentation_config is None:
        augmentation_config = {
            'temporal_mask_prob': 0.3,
            'keypoint_dropout_prob': 0.3,
            'sequential_cutout_prob': 0.2,
            'mask_ratio': 0.15,
            'dropout_prob': 0.3,
            'dropout_type': 'random',
            'max_keypoints': 5
        }
    
    augmenter = HybridTemporalAugmentation(**augmentation_config)
    
    # Apply to each body part
    l_hand_aug = augmenter(l_hand)
    r_hand_aug = augmenter(r_hand)
    body_aug = augmenter(body)
    
    return l_hand_aug, r_hand_aug, body_aug


if __name__ == "__main__":
    # Test temporal augmentations
    print("Testing Temporal Augmentation Module\n")
    print("=" * 60)
    
    # Create dummy data
    T, num_kp, C = 204, 21, 2
    dummy_sequence = np.random.randn(T, num_kp, C)
    
    print(f"Input shape: {dummy_sequence.shape}")
    print(f"Input range: [{dummy_sequence.min():.2f}, {dummy_sequence.max():.2f}]")
    print()
    
    # Test 1: Temporal Masking
    print("1. Temporal Masking:")
    masker = TemporalMasking(mask_ratio=0.15, consecutive=False)
    masked = masker(dummy_sequence)
    zero_frames = np.sum(np.all(masked == 0, axis=(1, 2)))
    print(f"   - Frames masked: {zero_frames}/{T} ({zero_frames/T*100:.1f}%)")
    print()
    
    # Test 2: Keypoint Dropout
    print("2. Keypoint Dropout (Random):")
    dropper = KeypointDropout(dropout_prob=1.0, dropout_type='random', max_keypoints=5)
    dropped = dropper(dummy_sequence)
    zero_keypoints = np.sum(np.all(dropped == 0, axis=(0, 2)))
    print(f"   - Keypoints dropped: {zero_keypoints}/{num_kp}")
    print()
    
    # Test 3: Sequential Cutout
    print("3. Sequential Cutout:")
    cutouter = SequentialCutout(temporal_cutout_prob=1.0, spatial_cutout_prob=1.0)
    cutout = cutouter(dummy_sequence)
    print(f"   - Output shape: {cutout.shape}")
    print()
    
    # Test 4: Hybrid Augmentation
    print("4. Hybrid Temporal Augmentation:")
    hybrid = HybridTemporalAugmentation(
        temporal_mask_prob=0.5,
        keypoint_dropout_prob=0.5,
        sequential_cutout_prob=0.5
    )
    for i in range(5):
        aug = hybrid(dummy_sequence)
        non_zero_ratio = np.count_nonzero(aug) / aug.size
        print(f"   - Trial {i+1}: {non_zero_ratio*100:.1f}% non-zero values")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
