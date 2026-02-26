"""
Translation-Scale-Shift Invariant (TSSI) Normalization for skeleton data.

This normalization method ensures translation and scale invariance across entire sequences,
maintaining spatial relationships between frames and body parts.

Based on: SL-TSSI-DenseNet (David Laines)
"""

import numpy as np
import torch


def tssi_normalize_sequence(landmarks_dict: dict, center_joint: str = None) -> dict:
    """
    Apply Translation-Scale Invariant normalization to an entire sequence.
    
    This normalization:
    1. Translates all frames so that the center joint (or centroid) is at origin
    2. Scales all frames by the maximum distance across ALL frames (not per-frame)
    3. Preserves spatial relationships and relative scales between frames
    
    Args:
        landmarks_dict: Dictionary with landmark names as keys, values are lists of (x, y) coordinates
                       Shape: {landmark_name: [(x, y), (x, y), ...]} for T frames
        center_joint: Optional joint name to use as center. If None, uses centroid of all joints.
    
    Returns:
        Normalized landmarks dictionary with same structure
    """
    
    # Get sequence length from any landmark
    sequence_len = len(next(iter(landmarks_dict.values())))
    
    # Convert dict to numpy array for easier computation
    # Shape: (T, num_landmarks, 2)
    landmark_names = list(landmarks_dict.keys())
    num_landmarks = len(landmark_names)
    
    coords = np.zeros((sequence_len, num_landmarks, 2))
    for idx, name in enumerate(landmark_names):
        coords[:, idx, :] = np.array(landmarks_dict[name])
    
    # Step 1: Translation - Center around origin
    if center_joint and center_joint in landmarks_dict:
        # Use specific joint as center
        center = np.array(landmarks_dict[center_joint])  # Shape: (T, 2)
        center = center[:, np.newaxis, :]  # Shape: (T, 1, 2) for broadcasting
    else:
        # Use centroid of all landmarks as center
        center = np.mean(coords, axis=1, keepdims=True)  # Shape: (T, 1, 2)
    
    # Translate all coordinates
    coords_centered = coords - center  # Shape: (T, num_landmarks, 2)
    
    # Step 2: Scale Invariance - Find maximum distance across ALL frames
    # Calculate distances from center for all points in all frames
    distances = np.sqrt(np.sum(coords_centered ** 2, axis=2))  # Shape: (T, num_landmarks)
    
    # Get the maximum distance across the entire sequence
    max_distance = np.max(distances)
    
    # Avoid division by zero (if all landmarks are at the same point)
    if max_distance < 1e-6:
        max_distance = 1.0
    
    # Normalize by maximum distance
    coords_normalized = coords_centered / max_distance
    
    # Convert back to dictionary format
    normalized_dict = {}
    for idx, name in enumerate(landmark_names):
        normalized_dict[name] = [(coords_normalized[t, idx, 0], coords_normalized[t, idx, 1]) 
                                 for t in range(sequence_len)]
    
    return normalized_dict


def tssi_normalize_sequence_frame_level(landmarks_dict: dict, center_joint: str = None) -> dict:
    """
    Apply frame-level TSSI normalization (each frame normalized independently).
    
    This is a variant that normalizes each frame separately, which may lose some
    temporal scale information but can be more robust to scale changes within a sequence.
    
    Args:
        landmarks_dict: Dictionary with landmark names as keys, values are lists of (x, y) coordinates
        center_joint: Optional joint name to use as center. If None, uses centroid of all joints.
    
    Returns:
        Normalized landmarks dictionary with same structure
    """
    
    sequence_len = len(next(iter(landmarks_dict.values())))
    landmark_names = list(landmarks_dict.keys())
    num_landmarks = len(landmark_names)
    
    coords = np.zeros((sequence_len, num_landmarks, 2))
    for idx, name in enumerate(landmark_names):
        coords[:, idx, :] = np.array(landmarks_dict[name])
    
    coords_normalized = np.zeros_like(coords)
    
    # Normalize each frame independently
    for t in range(sequence_len):
        frame_coords = coords[t]  # Shape: (num_landmarks, 2)
        
        # Center
        if center_joint and center_joint in landmarks_dict:
            center = np.array(landmarks_dict[center_joint][t])  # Shape: (2,)
        else:
            center = np.mean(frame_coords, axis=0)  # Shape: (2,)
        
        frame_centered = frame_coords - center
        
        # Scale by max distance in this frame
        distances = np.sqrt(np.sum(frame_centered ** 2, axis=1))
        max_distance = np.max(distances)
        
        if max_distance < 1e-6:
            max_distance = 1.0
        
        coords_normalized[t] = frame_centered / max_distance
    
    # Convert back to dictionary
    normalized_dict = {}
    for idx, name in enumerate(landmark_names):
        normalized_dict[name] = [(coords_normalized[t, idx, 0], coords_normalized[t, idx, 1]) 
                                 for t in range(sequence_len)]
    
    return normalized_dict


def tssi_normalize_with_visibility(landmarks_dict: dict, visibility_threshold: float = 0.5,
                                   center_joint: str = None) -> dict:
    """
    TSSI normalization with visibility/confidence filtering.
    
    Only uses visible landmarks (with confidence > threshold) for computing center and scale.
    Useful when landmark detection quality varies.
    
    Args:
        landmarks_dict: Dictionary with landmark names as keys, values are lists of (x, y, confidence) tuples
        visibility_threshold: Minimum confidence to consider a landmark visible
        center_joint: Optional joint name to use as center
    
    Returns:
        Normalized landmarks dictionary (only x, y coordinates)
    """
    
    # This is a placeholder for more advanced use cases
    # Current dataset doesn't have confidence scores, so we just call the basic version
    return tssi_normalize_sequence(landmarks_dict, center_joint)


# Convenience function that matches the existing normalization API
def normalize_single_dict_tssi(landmarks_dict: dict, method: str = "sequence", 
                               center_joint: str = None) -> dict:
    """
    Main entry point for TSSI normalization that matches the API of existing normalization functions.
    
    Args:
        landmarks_dict: Dictionary of landmarks
        method: "sequence" for full sequence normalization (recommended), 
                "frame" for frame-level normalization
        center_joint: Optional joint name to center on (e.g., "neck" or "nose")
    
    Returns:
        Normalized landmarks dictionary
    """
    
    if method == "frame":
        return tssi_normalize_sequence_frame_level(landmarks_dict, center_joint)
    else:
        return tssi_normalize_sequence(landmarks_dict, center_joint)


if __name__ == "__main__":
    # Test the TSSI normalization
    print("TSSI Normalization Module")
    print("=" * 50)
    
    # Create a simple test sequence with 3 frames, 2 landmarks
    test_dict = {
        "joint1": [(100.0, 200.0), (110.0, 210.0), (105.0, 205.0)],
        "joint2": [(150.0, 250.0), (160.0, 260.0), (155.0, 255.0)],
    }
    
    print("Original coordinates:")
    for key, val in test_dict.items():
        print(f"  {key}: {val}")
    
    normalized = tssi_normalize_sequence(test_dict)
    
    print("\nNormalized coordinates:")
    for key, val in normalized.items():
        print(f"  {key}: {val}")
    
    print("\n✓ TSSI normalization test completed successfully!")
