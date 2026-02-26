"""
Test script to verify default augmentation settings.
This helps confirm that skeleton augmentation is enabled by default.
"""

import argparse
import sys

# Add the parent directory to the path
sys.path.insert(0, '.')

from train import get_default_args


def test_default_values():
    print("="*60)
    print("TESTING DEFAULT AUGMENTATION VALUES")
    print("="*60)
    
    # Get default parser
    parser = get_default_args()
    args = parser.parse_args([])  # Parse with no arguments (all defaults)
    
    # Check skeleton augmentation defaults
    print("\n✓ Skeleton Augmentation Settings:")
    print(f"  - use_skeleton_augmentation: {args.use_skeleton_augmentation}")
    print(f"  - skeleton_aug_prob: {args.skeleton_aug_prob}")
    print(f"  - rotation_range: {args.rotation_range}")
    print(f"  - scale_range: {args.scale_range}")
    
    # Verify expected values
    assert args.use_skeleton_augmentation == True, "❌ use_skeleton_augmentation should be True by default!"
    assert args.skeleton_aug_prob == 0.5, "❌ skeleton_aug_prob should be 0.5 by default!"
    assert args.rotation_range == "-15,15", "❌ rotation_range should be '-15,15' by default!"
    assert args.scale_range == "0.8,1.2", "❌ scale_range should be '0.8,1.2' by default!"
    
    print("\n✓ All default values are correct!")
    
    # Parse the ranges
    rotation_range = tuple(map(float, args.rotation_range.split(',')))
    scale_range = tuple(map(float, args.scale_range.split(',')))
    
    print("\n✓ Parsed ranges:")
    print(f"  - rotation_range: {rotation_range} (type: {type(rotation_range)})")
    print(f"  - scale_range: {scale_range} (type: {type(scale_range)})")
    
    assert rotation_range == (-15.0, 15.0), "❌ Parsed rotation_range incorrect!"
    assert scale_range == (0.8, 1.2), "❌ Parsed scale_range incorrect!"
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nWhen you run training without arguments, skeleton augmentation")
    print("will be ENABLED by default with these settings:")
    print(f"  • Rotation: ±15 degrees")
    print(f"  • Scale: 0.8x - 1.2x")
    print(f"  • Probability: 50% per sample")
    print(f"  • Center: neck joint")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_default_values()
