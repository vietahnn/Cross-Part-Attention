"""
RandomSpeed (Time Warping) augmentation inspired by SL-TSSI-DenseNet.

This augmentation changes the temporal dimension (number of frames) of a sequence,
simulating different signing speeds or video recording frame rates.
"""

import random
import numpy as np
from scipy import interpolate
from typing import Dict, Tuple


def augment_random_speed(sign: dict, min_frames: int = None, max_frames: int = None,
                         speed_range: Tuple[float, float] = (0.8, 1.2),
                         prob: float = 1.0) -> dict:
    """
    RandomSpeed augmentation - changes the temporal dimension of the sequence.
    
    This simulates different signing speeds by interpolating frames to create
    a longer (slower) or shorter (faster) sequence.
    
    Can be specified either by:
    1. Absolute frame counts: min_frames / max_frames
    2. Relative speed factor: speed_range (e.g., 0.8 = 80% speed = slower)
    
    Args:
        sign: Dictionary with sequential skeletal data {landmark: [(x, y), ...]}
        min_frames: Minimum number of frames (absolute). If None, uses speed_range.
        max_frames: Maximum number of frames (absolute). If None, uses speed_range.
        speed_range: Relative speed range as tuple (min_speed, max_speed).
                    1.0 = original speed, <1.0 = slower (more frames), >1.0 = faster (fewer frames)
        prob: Probability to apply this augmentation (default: 1.0)
    
    Returns:
        Dictionary with time-warped skeletal data (different number of frames)
    """
    
    if random.random() > prob:
        return sign
    
    # Get current sequence length
    current_frames = len(next(iter(sign.values())))
    
    # Determine target number of frames
    if min_frames is not None and max_frames is not None:
        # Use absolute frame counts
        actual_min = min(current_frames, min_frames)
        actual_max = max_frames
        target_frames = random.randint(actual_min, actual_max)
    else:
        # Use relative speed factor
        min_speed, max_speed = speed_range
        speed_factor = random.uniform(min_speed, max_speed)
        # Note: lower speed = more frames (slower motion)
        # speed 0.8 = 80% speed = 1/0.8 = 1.25x frames
        target_frames = max(1, int(current_frames / speed_factor))
    
    # If target is same as current, no change needed
    if target_frames == current_frames:
        return sign
    
    # Interpolate each landmark's trajectory
    warped_sign = {}
    
    for landmark_name, coordinates in sign.items():
        # Extract x and y coordinates
        x_coords = np.array([coord[0] for coord in coordinates])
        y_coords = np.array([coord[1] for coord in coordinates])
        
        # Create interpolation functions
        # Use cubic interpolation for smooth motion
        old_indices = np.arange(current_frames)
        new_indices = np.linspace(0, current_frames - 1, target_frames)
        
        # Interpolate x and y separately
        try:
            # Try cubic interpolation first (smoother)
            if current_frames >= 4:  # Need at least 4 points for cubic
                interp_x = interpolate.interp1d(old_indices, x_coords, kind='cubic', 
                                               fill_value='extrapolate')
                interp_y = interpolate.interp1d(old_indices, y_coords, kind='cubic',
                                               fill_value='extrapolate')
            else:
                # Fall back to linear for short sequences
                interp_x = interpolate.interp1d(old_indices, x_coords, kind='linear',
                                               fill_value='extrapolate')
                interp_y = interpolate.interp1d(old_indices, y_coords, kind='linear',
                                               fill_value='extrapolate')
            
            new_x = interp_x(new_indices)
            new_y = interp_y(new_indices)
        except Exception as e:
            # If interpolation fails, fall back to simple resampling
            indices = np.round(new_indices).astype(int)
            indices = np.clip(indices, 0, current_frames - 1)
            new_x = x_coords[indices]
            new_y = y_coords[indices]
        
        # Convert back to list of tuples
        warped_sign[landmark_name] = [(float(x), float(y)) for x, y in zip(new_x, new_y)]
    
    return warped_sign


def augment_random_speed_uniform(sign: dict, 
                                 target_frames: int = None,
                                 frame_range: Tuple[int, int] = (40, 128),
                                 prob: float = 1.0) -> dict:
    """
    Uniform RandomSpeed - resample to a specific or random target frame count.
    
    This version is simpler and ensures all sequences have similar lengths,
    which can be useful for batching.
    
    Args:
        sign: Dictionary with sequential skeletal data
        target_frames: Specific target frame count. If None, random from frame_range.
        frame_range: Range to randomly sample target frames from (min, max)
        prob: Probability to apply augmentation
    
    Returns:
        Dictionary with resampled skeletal data
    """
    
    if random.random() > prob:
        return sign
    
    # Get current sequence length
    current_frames = len(next(iter(sign.values())))
    
    # Determine target frames
    if target_frames is None:
        min_frames, max_frames = frame_range
        actual_min = min(current_frames, min_frames)
        target_frames = random.randint(actual_min, max_frames)
    
    if target_frames == current_frames:
        return sign
    
    # Resample using interpolation
    return _interpolate_sequence(sign, current_frames, target_frames)


def augment_random_speed_fast(sign: dict,
                              speed_range: Tuple[float, float] = (0.8, 1.2),
                              prob: float = 1.0) -> dict:
    """
    Fast RandomSpeed using simple linear interpolation.
    
    This is a faster version that uses numpy's built-in interpolation
    instead of scipy, trading some smoothness for speed.
    
    Args:
        sign: Dictionary with sequential skeletal data
        speed_range: Relative speed range (min_speed, max_speed)
        prob: Probability to apply augmentation
    
    Returns:
        Dictionary with time-warped skeletal data
    """
    
    if random.random() > prob:
        return sign
    
    current_frames = len(next(iter(sign.values())))
    
    # Calculate target frames
    min_speed, max_speed = speed_range
    speed_factor = random.uniform(min_speed, max_speed)
    target_frames = max(1, int(current_frames / speed_factor))
    
    if target_frames == current_frames:
        return sign
    
    # Fast linear interpolation
    warped_sign = {}
    old_indices = np.arange(current_frames)
    new_indices = np.linspace(0, current_frames - 1, target_frames)
    
    for landmark_name, coordinates in sign.items():
        x_coords = np.array([coord[0] for coord in coordinates])
        y_coords = np.array([coord[1] for coord in coordinates])
        
        # Simple linear interpolation using numpy
        new_x = np.interp(new_indices, old_indices, x_coords)
        new_y = np.interp(new_indices, old_indices, y_coords)
        
        warped_sign[landmark_name] = [(float(x), float(y)) for x, y in zip(new_x, new_y)]
    
    return warped_sign


def _interpolate_sequence(sign: dict, current_frames: int, target_frames: int) -> dict:
    """
    Helper function to interpolate a sequence to a target number of frames.
    """
    
    warped_sign = {}
    old_indices = np.arange(current_frames)
    new_indices = np.linspace(0, current_frames - 1, target_frames)
    
    for landmark_name, coordinates in sign.items():
        x_coords = np.array([coord[0] for coord in coordinates])
        y_coords = np.array([coord[1] for coord in coordinates])
        
        # Use scipy interpolation for better quality
        if current_frames >= 4:
            interp_x = interpolate.interp1d(old_indices, x_coords, kind='cubic',
                                           fill_value='extrapolate')
            interp_y = interpolate.interp1d(old_indices, y_coords, kind='cubic',
                                           fill_value='extrapolate')
        else:
            interp_x = interpolate.interp1d(old_indices, x_coords, kind='linear',
                                           fill_value='extrapolate')
            interp_y = interpolate.interp1d(old_indices, y_coords, kind='linear',
                                           fill_value='extrapolate')
        
        new_x = interp_x(new_indices)
        new_y = interp_y(new_indices)
        
        warped_sign[landmark_name] = [(float(x), float(y)) for x, y in zip(new_x, new_y)]
    
    return warped_sign


if __name__ == "__main__":
    print("RandomSpeed (Time Warping) augmentation module")
    print("=" * 60)
    print("Available functions:")
    print("  - augment_random_speed: Main function with speed_range")
    print("  - augment_random_speed_uniform: Uniform target frame count")
    print("  - augment_random_speed_fast: Fast linear interpolation")
    print("\n✓ Inspired by SL-TSSI-DenseNet temporal augmentation")
    print("=" * 60)
    
    # Simple test
    test_sign = {
        "joint1": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
        "joint2": [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)],
    }
    
    print(f"\nOriginal sequence: {len(test_sign['joint1'])} frames")
    
    # Test slower (more frames)
    slower = augment_random_speed(test_sign, speed_range=(0.5, 0.5))
    print(f"Slower (0.5x speed): {len(slower['joint1'])} frames")
    
    # Test faster (fewer frames)
    faster = augment_random_speed(test_sign, speed_range=(2.0, 2.0))
    print(f"Faster (2.0x speed): {len(faster['joint1'])} frames")
    
    print("\n✓ RandomSpeed augmentation test completed!")
