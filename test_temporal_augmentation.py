"""
Test script for Temporal Augmentation Module

This script validates that temporal masking and keypoint dropout work correctly
with the Siformer training pipeline.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from augmentations.temporal_augmentations import (
    TemporalMasking,
    KeypointDropout,
    SequentialCutout,
    HybridTemporalAugmentation,
    apply_temporal_augmentation
)


def test_temporal_masking():
    """Test temporal masking functionality"""
    print("\n" + "="*70)
    print("TEST 1: Temporal Masking")
    print("="*70)
    
    # Create dummy sequence: (T=204 frames, 21 keypoints, 2 coords)
    T, num_kp, C = 204, 21, 2
    sequence = np.random.randn(T, num_kp, C) + 5.0  # Non-zero data
    
    # Test with different mask ratios
    for mask_ratio in [0.1, 0.15, 0.2]:
        masker = TemporalMasking(mask_ratio=mask_ratio, consecutive=False)
        masked = masker(sequence)
        
        # Count masked frames
        zero_frames = np.sum(np.all(masked == 0, axis=(1, 2)))
        expected_ratio = zero_frames / T
        
        print(f"Mask Ratio {mask_ratio:.2f}:")
        print(f"  - Frames masked: {zero_frames}/{T} ({expected_ratio*100:.1f}%)")
        print(f"  - Original non-zero: {np.count_nonzero(sequence)}")
        print(f"  - Masked non-zero: {np.count_nonzero(masked)}")
        
        assert zero_frames <= T * mask_ratio * 1.5, "Too many frames masked!"
    
    print("✓ Temporal Masking passed!")


def test_keypoint_dropout():
    """Test keypoint dropout functionality"""
    print("\n" + "="*70)
    print("TEST 2: Keypoint Dropout")
    print("="*70)
    
    # Create dummy sequence
    T, num_kp, C = 204, 21, 2
    sequence = np.ones((T, num_kp, C)) * 3.0
    
    # Test random dropout
    print("\nRandom Keypoint Dropout:")
    dropper = KeypointDropout(dropout_prob=1.0, dropout_type='random', max_keypoints=5)
    dropped = dropper(sequence)
    
    # Check which keypoints were dropped
    zero_keypoints_mask = np.all(dropped == 0, axis=(0, 2))
    num_dropped = np.sum(zero_keypoints_mask)
    
    print(f"  - Keypoints dropped: {num_dropped}/{num_kp}")
    print(f"  - Dropped indices: {np.where(zero_keypoints_mask)[0].tolist()}")
    
    assert 1 <= num_dropped <= 5, f"Expected 1-5 keypoints dropped, got {num_dropped}"
    
    # Test bodypart dropout
    print("\nBodypart Dropout:")
    body_parts = {
        'left_hand': list(range(0, 21)),
        'right_hand': list(range(21, 42)),
        'body': list(range(42, 54))
    }
    dropper_bodypart = KeypointDropout(
        dropout_prob=1.0, 
        dropout_type='bodypart',
        body_parts_config=body_parts
    )
    
    # Test on larger sequence
    T, num_kp, C = 204, 54, 2  # Full body
    sequence_full = np.ones((T, num_kp, C)) * 3.0
    dropped_full = dropper_bodypart(sequence_full)
    
    zero_keypoints = np.sum(np.all(dropped_full == 0, axis=(0, 2)))
    print(f"  - Keypoints zeroed: {zero_keypoints}/{num_kp}")
    
    print("✓ Keypoint Dropout passed!")


def test_sequential_cutout():
    """Test sequential cutout functionality"""
    print("\n" + "="*70)
    print("TEST 3: Sequential Cutout")
    print("="*70)
    
    T, num_kp, C = 204, 21, 2
    sequence = np.random.randn(T, num_kp, C) + 5.0
    
    cutouter = SequentialCutout(
        temporal_cutout_prob=1.0,
        spatial_cutout_prob=1.0,
        temporal_cutout_frames=15,
        spatial_cutout_keypoints=5
    )
    
    cutout = cutouter(sequence)
    
    # Check temporal cutout
    zero_frames = np.sum(np.all(cutout == 0, axis=(1, 2)))
    print(f"Temporal cutout:")
    print(f"  - Zero frames: {zero_frames}/{T}")
    
    # Check spatial cutout
    zero_keypoints = []
    for kp_idx in range(num_kp):
        if np.all(cutout[:, kp_idx, :] == 0):
            zero_keypoints.append(kp_idx)
    
    print(f"Spatial cutout:")
    print(f"  - Zero keypoints across all frames: {len(zero_keypoints)}")
    
    print("✓ Sequential Cutout passed!")


def test_hybrid_augmentation():
    """Test hybrid augmentation"""
    print("\n" + "="*70)
    print("TEST 4: Hybrid Temporal Augmentation")
    print("="*70)
    
    T, num_kp, C = 204, 21, 2
    sequence = np.random.randn(T, num_kp, C) + 5.0
    
    hybrid = HybridTemporalAugmentation(
        temporal_mask_prob=0.5,
        keypoint_dropout_prob=0.5,
        sequential_cutout_prob=0.5,
        mask_ratio=0.15,
        dropout_prob=0.3
    )
    
    print("Running 10 augmentation trials:")
    non_zero_ratios = []
    for i in range(10):
        aug = hybrid(sequence)
        non_zero_ratio = np.count_nonzero(aug) / aug.size
        non_zero_ratios.append(non_zero_ratio)
        print(f"  Trial {i+1}: {non_zero_ratio*100:.1f}% non-zero")
    
    avg_ratio = np.mean(non_zero_ratios)
    std_ratio = np.std(non_zero_ratios)
    
    print(f"\nStatistics:")
    print(f"  - Average non-zero: {avg_ratio*100:.1f}% ± {std_ratio*100:.1f}%")
    print(f"  - Min: {min(non_zero_ratios)*100:.1f}%")
    print(f"  - Max: {max(non_zero_ratios)*100:.1f}%")
    
    assert avg_ratio > 0.6, "Too much data being zeroed out!"
    
    print("✓ Hybrid Augmentation passed!")


def test_integration_with_sign_data():
    """Test with sign language-like data structure"""
    print("\n" + "="*70)
    print("TEST 5: Integration with Sign Language Data")
    print("="*70)
    
    # Simulate sign language data
    T = 204  # frames
    l_hand = np.random.randn(T, 21, 2) + 0.3  # Left hand
    r_hand = np.random.randn(T, 21, 2) - 0.2  # Right hand
    body = np.random.randn(T, 12, 2) + 0.1    # Body
    
    print(f"Input shapes:")
    print(f"  - Left hand: {l_hand.shape}")
    print(f"  - Right hand: {r_hand.shape}")
    print(f"  - Body: {body.shape}")
    
    # Apply augmentation
    config = {
        'temporal_mask_prob': 0.3,
        'keypoint_dropout_prob': 0.3,
        'sequential_cutout_prob': 0.2,
        'mask_ratio': 0.15,
        'dropout_prob': 0.3,
        'dropout_type': 'random',
        'max_keypoints': 5
    }
    
    l_hand_aug, r_hand_aug, body_aug = apply_temporal_augmentation(
        l_hand, r_hand, body, config
    )
    
    print(f"\nOutput shapes:")
    print(f"  - Left hand: {l_hand_aug.shape}")
    print(f"  - Right hand: {r_hand_aug.shape}")
    print(f"  - Body: {body_aug.shape}")
    
    print(f"\nData preservation:")
    print(f"  - Left hand: {np.count_nonzero(l_hand_aug)/l_hand_aug.size*100:.1f}% non-zero")
    print(f"  - Right hand: {np.count_nonzero(r_hand_aug)/r_hand_aug.size*100:.1f}% non-zero")
    print(f"  - Body: {np.count_nonzero(body_aug)/body_aug.size*100:.1f}% non-zero")
    
    assert l_hand_aug.shape == l_hand.shape, "Shape mismatch!"
    assert r_hand_aug.shape == r_hand.shape, "Shape mismatch!"
    assert body_aug.shape == body.shape, "Shape mismatch!"
    
    print("✓ Integration test passed!")


def test_torch_compatibility():
    """Test compatibility with PyTorch tensors"""
    print("\n" + "="*70)
    print("TEST 6: PyTorch Tensor Compatibility")
    print("="*70)
    
    # Create PyTorch tensor
    T, num_kp, C = 204, 21, 2
    tensor = torch.randn(T, num_kp, C) + 2.0
    
    print(f"Input: PyTorch tensor with shape {tensor.shape}")
    
    # Convert to numpy, augment, convert back
    numpy_array = tensor.numpy()
    
    hybrid = HybridTemporalAugmentation(
        temporal_mask_prob=0.5,
        keypoint_dropout_prob=0.5,
        mask_ratio=0.15
    )
    
    augmented_numpy = hybrid(numpy_array)
    augmented_tensor = torch.from_numpy(augmented_numpy).float()
    
    print(f"Output: PyTorch tensor with shape {augmented_tensor.shape}")
    print(f"Output dtype: {augmented_tensor.dtype}")
    print(f"Data preserved: {torch.count_nonzero(augmented_tensor)/augmented_tensor.numel()*100:.1f}%")
    
    assert augmented_tensor.shape == tensor.shape, "Shape mismatch!"
    assert augmented_tensor.dtype == torch.float32, "Dtype mismatch!"
    
    print("✓ PyTorch compatibility passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "TEMPORAL AUGMENTATION TEST SUITE" + " "*21 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        test_temporal_masking()
        test_keypoint_dropout()
        test_sequential_cutout()
        test_hybrid_augmentation()
        test_integration_with_sign_data()
        test_torch_compatibility()
        
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + " "*20 + "ALL TESTS PASSED! ✓✓✓" + " "*27 + "█")
        print("█" + " "*68 + "█")
        print("█"*70 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
