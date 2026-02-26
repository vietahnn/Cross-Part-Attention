"""
Test script for comparing normalize v1 vs v2 (semi-isolated normalization)

This script demonstrates the difference between:
- V1: Standard bounding box normalization for hands
- V2: Semi-isolated normalization (relative to body reference point first, then normalized)

Usage:
    python test_normalize_v2.py

The v2 method is inspired by Teledeaf's semi-isolated normalization approach,
adapted for 2D coordinates used in Siformer.
"""

import numpy as np
from normalization.hand_normalization import normalize_single_dict, normalize_single_dict_v2

def create_sample_data():
    """Create sample hand and body data for testing"""
    # Sample data: 5 frames, with hand landmarks and neck position
    sample_data = {
        # Body reference point (neck)
        "neck": [
            [320, 200],  # Frame 1: center of image
            [330, 210],  # Frame 2: slightly moved
            [325, 205],  # Frame 3
            [320, 200],  # Frame 4
            [315, 195],  # Frame 5
        ],
        
        # Right hand (hand_index = 1) - near the body
        "wrist_1": [
            [400, 300],  # Frame 1
            [410, 310],  # Frame 2
            [405, 305],  # Frame 3
            [400, 300],  # Frame 4
            [395, 295],  # Frame 5
        ],
        "indexTip_1": [
            [420, 280],
            [430, 290],
            [425, 285],
            [420, 280],
            [415, 275],
        ],
        "thumbTip_1": [
            [410, 320],
            [420, 330],
            [415, 325],
            [410, 320],
            [405, 315],
        ],
        
        # Left hand (hand_index = 0) - farther from body
        "wrist_0": [
            [200, 300],  # Frame 1
            [210, 310],  # Frame 2
            [205, 305],  # Frame 3
            [200, 300],  # Frame 4
            [195, 295],  # Frame 5
        ],
        "indexTip_0": [
            [180, 280],
            [190, 290],
            [185, 285],
            [180, 280],
            [175, 275],
        ],
        "thumbTip_0": [
            [190, 320],
            [200, 330],
            [195, 325],
            [190, 320],
            [185, 315],
        ],
    }
    
    # Fill in remaining hand landmarks with interpolated values
    hand_identifiers = [
        "indexDIP", "indexPIP", "indexMCP",
        "middleTip", "middleDIP", "middlePIP", "middleMCP",
        "ringTip", "ringDIP", "ringPIP", "ringMCP",
        "littleTip", "littleDIP", "littlePIP", "littleMCP",
        "thumbIP", "thumbMP", "thumbCMC"
    ]
    
    for hand_idx in [0, 1]:
        wrist = sample_data[f"wrist_{hand_idx}"]
        for identifier in hand_identifiers:
            key = f"{identifier}_{hand_idx}"
            # Simple interpolation: random offset from wrist
            sample_data[key] = [
                [w[0] + np.random.randint(-30, 30), w[1] + np.random.randint(-30, 30)]
                for w in wrist
            ]
    
    return sample_data


def compare_normalization_methods():
    """Compare v1 and v2 normalization methods"""
    
    print("=" * 80)
    print("COMPARING NORMALIZATION METHODS: V1 vs V2")
    print("=" * 80)
    print()
    
    # Create sample data
    data = create_sample_data()
    
    # Make copies for each normalization method
    import copy
    data_v1 = copy.deepcopy(data)
    data_v2 = copy.deepcopy(data)
    
    print("Original data (Frame 1):")
    print(f"  Neck:        {data['neck'][0]}")
    print(f"  Left wrist:  {data['wrist_0'][0]}")
    print(f"  Right wrist: {data['wrist_1'][0]}")
    print()
    
    # Apply V1 normalization
    print("-" * 80)
    print("Applying V1 normalization (standard bounding box)...")
    data_v1_normalized = normalize_single_dict(data_v1)
    
    print("V1 Normalized (Frame 1):")
    print(f"  Left wrist:  {data_v1_normalized['wrist_0'][0]}")
    print(f"  Right wrist: {data_v1_normalized['wrist_1'][0]}")
    print()
    
    # Apply V2 normalization
    print("-" * 80)
    print("Applying V2 normalization (semi-isolated, relative to neck)...")
    data_v2_normalized = normalize_single_dict_v2(data_v2, body_ref_key="neck")
    
    print("V2 Normalized (Frame 1):")
    print(f"  Left wrist:  {data_v2_normalized['wrist_0'][0]}")
    print(f"  Right wrist: {data_v2_normalized['wrist_1'][0]}")
    print()
    
    # Analyze differences
    print("=" * 80)
    print("ANALYSIS: Key Differences")
    print("=" * 80)
    print()
    print("V1 (Standard):")
    print("  - Normalizes hands independently using their own bounding box")
    print("  - Hand position in image space is lost after normalization")
    print("  - Focus on hand shape only")
    print()
    print("V2 (Semi-isolated):")
    print("  - First converts coordinates relative to body reference (neck)")
    print("  - Then normalizes within hand's bounding box")
    print("  - Preserves spatial relationship between hands and body")
    print("  - Potentially better for sign language where hand-body relation matters")
    print()
    
    # Compare some landmarks across frames
    print("-" * 80)
    print("Comparison across 3 frames:")
    print()
    
    for frame_idx in [0, 1, 2]:
        print(f"Frame {frame_idx + 1}:")
        print(f"  V1 - Left wrist:  {data_v1_normalized['wrist_0'][frame_idx]}")
        print(f"  V2 - Left wrist:  {data_v2_normalized['wrist_0'][frame_idx]}")
        print()
    
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    compare_normalization_methods()
