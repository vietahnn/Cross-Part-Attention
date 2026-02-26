"""
Test script to verify default RandomSpeed augmentation settings.
This helps confirm that time warping is enabled by default.
"""

import argparse
import sys

# Add the parent directory to the path
sys.path.insert(0, '.')

from train import get_default_args


def test_default_values():
    print("="*60)
    print("TESTING DEFAULT RANDOMSPEED AUGMENTATION VALUES")
    print("="*60)
    
    # Get default parser
    parser = get_default_args()
    args = parser.parse_args([])  # Parse with no arguments (all defaults)
    
    # Check RandomSpeed augmentation defaults
    print("\n✓ RandomSpeed (Time Warping) Settings:")
    print(f"  - use_random_speed: {args.use_random_speed}")
    print(f"  - speed_aug_prob: {args.speed_aug_prob}")
    print(f"  - speed_range: {args.speed_range}")
    
    # Verify expected values
    assert args.use_random_speed == True, "❌ use_random_speed should be True by default!"
    assert args.speed_aug_prob == 0.5, "❌ speed_aug_prob should be 0.5 by default!"
    assert args.speed_range == "0.8,1.2", "❌ speed_range should be '0.8,1.2' by default!"
    
    print("\n✓ All default values are correct!")
    
    # Parse the ranges
    speed_range = tuple(map(float, args.speed_range.split(',')))
    
    print("\n✓ Parsed ranges:")
    print(f"  - speed_range: {speed_range} (type: {type(speed_range)})")
    
    assert speed_range == (0.8, 1.2), "❌ Parsed speed_range incorrect!"
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nWhen you run training without arguments, RandomSpeed augmentation")
    print("will be ENABLED by default with these settings:")
    print(f"  • Speed range: 0.8x - 1.2x")
    print(f"  • Probability: 50% per sample")
    print(f"  • Effect: Simulates different signing speeds (time warping)")
    print(f"  • Implementation: Fast linear interpolation")
    print("="*60 + "\n")


def test_temporal_augmentation():
    """Test the actual temporal augmentation module"""
    print("\n" + "="*60)
    print("TESTING TEMPORAL AUGMENTATION MODULE")
    print("="*60)
    
    from augmentations.temporal_augmentation import augment_random_speed_fast
    
    # Create test sequence
    test_sign = {
        "joint1": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
        "joint2": [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)],
    }
    
    original_length = len(test_sign['joint1'])
    print(f"\n✓ Original sequence: {original_length} frames")
    
    # Test slower (more frames) - speed 0.8 means 1/0.8 = 1.25x frames
    slower = augment_random_speed_fast(test_sign, speed_range=(0.8, 0.8), prob=1.0)
    print(f"✓ Slower (0.8x speed): {len(slower['joint1'])} frames (expected ~6)")
    
    # Test faster (fewer frames) - speed 1.2 means 1/1.2 = 0.83x frames
    faster = augment_random_speed_fast(test_sign, speed_range=(1.2, 1.2), prob=1.0)
    print(f"✓ Faster (1.2x speed): {len(faster['joint1'])} frames (expected ~4)")
    
    # Test no change
    normal = augment_random_speed_fast(test_sign, speed_range=(1.0, 1.0), prob=1.0)
    print(f"✓ Normal (1.0x speed): {len(normal['joint1'])} frames (expected 5)")
    
    # Test probability
    no_change = augment_random_speed_fast(test_sign, speed_range=(0.5, 0.5), prob=0.0)
    assert len(no_change['joint1']) == original_length, "❌ Zero probability should return original!"
    print(f"✓ Zero probability: {len(no_change['joint1'])} frames (unchanged)")
    
    print("\n" + "="*60)
    print("✅ TEMPORAL AUGMENTATION TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_default_values()
    test_temporal_augmentation()
