"""
Skeleton-aware augmentation techniques inspired by SL-TSSI-DenseNet.

These augmentations work directly on skeleton coordinates and maintain
anatomical validity, unlike pixel-based image augmentations.
"""

import math
import random
import numpy as np
from typing import Dict, Tuple, List


def augment_random_rotation(sign: dict, angle_range: tuple = (-15, 15), 
                            center: str = "auto", prob: float = 1.0) -> dict:
    """
    RandomRotation augmentation inspired by SL-TSSI-DenseNet.
    
    Rotates all skeleton landmarks around a center point by a random angle.
    This simulates different camera viewpoints or person orientations.
    
    Unlike the default augment_rotate which uses fixed center (0.5, 0.5),
    this uses the skeleton's centroid or a specific joint as rotation center.
    
    Args:
        sign: Dictionary with sequential skeletal data {landmark: [(x, y), ...]}
        angle_range: Tuple (min_angle, max_angle) in degrees (default: -15 to +15)
        center: Rotation center - "auto" (centroid), "neck", or specific joint name
        prob: Probability to apply this augmentation (default: 1.0)
    
    Returns:
        Dictionary with rotated skeletal data
    """
    
    if random.random() > prob:
        return sign
    
    # Get sequence length
    sequence_len = len(next(iter(sign.values())))
    
    # Choose random angle
    angle = math.radians(random.uniform(*angle_range))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Compute rotation center for each frame
    rotated_sign = {}
    
    for landmark_name, coordinates in sign.items():
        rotated_coords = []
        
        for frame_idx in range(sequence_len):
            # Compute center for this frame
            if center == "auto":
                # Use centroid of all visible landmarks in this frame
                frame_coords = [sign[key][frame_idx] for key in sign.keys()]
                cx = np.mean([coord[0] for coord in frame_coords])
                cy = np.mean([coord[1] for coord in frame_coords])
            elif center in sign:
                # Use specific joint
                cx, cy = sign[center][frame_idx]
            else:
                # Fallback to frame center
                cx, cy = 0.5, 0.5
            
            # Get current point
            px, py = coordinates[frame_idx]
            
            # Rotate around center
            qx = cx + cos_a * (px - cx) - sin_a * (py - cy)
            qy = cy + sin_a * (px - cx) + cos_a * (py - cy)
            
            rotated_coords.append((qx, qy))
        
        rotated_sign[landmark_name] = rotated_coords
    
    return rotated_sign


def augment_random_scale(sign: dict, scale_range: tuple = (0.8, 1.2),
                         center: str = "auto", prob: float = 1.0) -> dict:
    """
    RandomScale augmentation inspired by SL-TSSI-DenseNet.
    
    Scales the skeleton size uniformly by a random factor.
    This simulates different distances from the camera or person sizes.
    
    Args:
        sign: Dictionary with sequential skeletal data {landmark: [(x, y), ...]}
        scale_range: Tuple (min_scale, max_scale) (default: 0.8 to 1.2)
                    1.0 = original size, <1.0 = smaller, >1.0 = larger
        center: Scale center - "auto" (centroid), "neck", or specific joint name
        prob: Probability to apply this augmentation (default: 1.0)
    
    Returns:
        Dictionary with scaled skeletal data
    """
    
    if random.random() > prob:
        return sign
    
    # Get sequence length
    sequence_len = len(next(iter(sign.values())))
    
    # Choose random scale factor (same for entire sequence to preserve dynamics)
    scale_factor = random.uniform(*scale_range)
    
    # Compute scaling center for each frame
    scaled_sign = {}
    
    for landmark_name, coordinates in sign.items():
        scaled_coords = []
        
        for frame_idx in range(sequence_len):
            # Compute center for this frame
            if center == "auto":
                # Use centroid of all visible landmarks in this frame
                frame_coords = [sign[key][frame_idx] for key in sign.keys()]
                cx = np.mean([coord[0] for coord in frame_coords])
                cy = np.mean([coord[1] for coord in frame_coords])
            elif center in sign:
                # Use specific joint
                cx, cy = sign[center][frame_idx]
            else:
                # Fallback to frame center
                cx, cy = 0.5, 0.5
            
            # Get current point
            px, py = coordinates[frame_idx]
            
            # Scale around center
            qx = cx + scale_factor * (px - cx)
            qy = cy + scale_factor * (py - cy)
            
            scaled_coords.append((qx, qy))
        
        scaled_sign[landmark_name] = scaled_coords
    
    return scaled_sign


def augment_random_rotation_and_scale(sign: dict, 
                                     angle_range: tuple = (-15, 15),
                                     scale_range: tuple = (0.8, 1.2),
                                     center: str = "auto",
                                     prob: float = 1.0) -> dict:
    """
    Combined RandomRotation + RandomScale augmentation.
    
    Applies both rotation and scaling in a single pass for efficiency.
    This is the recommended way to use both augmentations together.
    
    Args:
        sign: Dictionary with sequential skeletal data
        angle_range: Rotation angle range in degrees
        scale_range: Scale factor range
        center: Transformation center
        prob: Probability to apply this augmentation
    
    Returns:
        Dictionary with transformed skeletal data
    """
    
    if random.random() > prob:
        return sign
    
    # Get sequence length
    sequence_len = len(next(iter(sign.values())))
    
    # Choose random angle and scale
    angle = math.radians(random.uniform(*angle_range))
    scale = random.uniform(*scale_range)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Apply combined transformation
    transformed_sign = {}
    
    for landmark_name, coordinates in sign.items():
        transformed_coords = []
        
        for frame_idx in range(sequence_len):
            # Compute center for this frame
            if center == "auto":
                frame_coords = [sign[key][frame_idx] for key in sign.keys()]
                cx = np.mean([coord[0] for coord in frame_coords])
                cy = np.mean([coord[1] for coord in frame_coords])
            elif center in sign:
                cx, cy = sign[center][frame_idx]
            else:
                cx, cy = 0.5, 0.5
            
            # Get current point
            px, py = coordinates[frame_idx]
            
            # Apply scale then rotation (order matters!)
            # First scale
            px_scaled = cx + scale * (px - cx)
            py_scaled = cy + scale * (py - cy)
            
            # Then rotate
            qx = cx + cos_a * (px_scaled - cx) - sin_a * (py_scaled - cy)
            qy = cy + sin_a * (px_scaled - cx) + cos_a * (py_scaled - cy)
            
            transformed_coords.append((qx, qy))
        
        transformed_sign[landmark_name] = transformed_coords
    
    return transformed_sign


def augment_random_skeleton_transform(sign: dict,
                                      rotation_prob: float = 0.5,
                                      scale_prob: float = 0.5,
                                      angle_range: tuple = (-15, 15),
                                      scale_range: tuple = (0.8, 1.2),
                                      center: str = "auto") -> dict:
    """
    Flexible skeleton transformation with independent probabilities.
    
    This allows for:
    - Only rotation (50% chance)
    - Only scaling (50% chance)
    - Both rotation and scaling (25% chance when both triggered)
    - No transformation (25% chance when neither triggered)
    
    Args:
        sign: Dictionary with sequential skeletal data
        rotation_prob: Probability to apply rotation (0.0 to 1.0)
        scale_prob: Probability to apply scaling (0.0 to 1.0)
        angle_range: Rotation angle range in degrees
        scale_range: Scale factor range
        center: Transformation center
    
    Returns:
        Dictionary with potentially transformed skeletal data
    """
    
    apply_rotation = random.random() < rotation_prob
    apply_scale = random.random() < scale_prob
    
    if not apply_rotation and not apply_scale:
        return sign
    
    # Get sequence length
    sequence_len = len(next(iter(sign.values())))
    
    # Prepare transformation parameters
    angle = math.radians(random.uniform(*angle_range)) if apply_rotation else 0.0
    scale = random.uniform(*scale_range) if apply_scale else 1.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Apply transformation
    transformed_sign = {}
    
    for landmark_name, coordinates in sign.items():
        transformed_coords = []
        
        for frame_idx in range(sequence_len):
            # Compute center
            if center == "auto":
                frame_coords = [sign[key][frame_idx] for key in sign.keys()]
                cx = np.mean([coord[0] for coord in frame_coords])
                cy = np.mean([coord[1] for coord in frame_coords])
            elif center in sign:
                cx, cy = sign[center][frame_idx]
            else:
                cx, cy = 0.5, 0.5
            
            px, py = coordinates[frame_idx]
            
            # Apply scale
            if apply_scale:
                px = cx + scale * (px - cx)
                py = cy + scale * (py - cy)
            
            # Apply rotation
            if apply_rotation:
                qx = cx + cos_a * (px - cx) - sin_a * (py - cy)
                qy = cy + sin_a * (px - cx) + cos_a * (py - cy)
            else:
                qx, qy = px, py
            
            transformed_coords.append((qx, qy))
        
        transformed_sign[landmark_name] = transformed_coords
    
    return transformed_sign


if __name__ == "__main__":
    print("Skeleton-aware augmentation module")
    print("=" * 60)
    print("Available augmentations:")
    print("  - augment_random_rotation: Rotate skeleton by random angle")
    print("  - augment_random_scale: Scale skeleton by random factor")
    print("  - augment_random_rotation_and_scale: Combined transformation")
    print("  - augment_random_skeleton_transform: Flexible with probabilities")
    print("\n✓ Inspired by SL-TSSI-DenseNet augmentation techniques")
