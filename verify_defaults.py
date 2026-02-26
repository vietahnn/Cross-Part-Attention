"""
Complete verification test: Ensure RandomSpeed works by default with NO arguments.
This simulates running: python train.py --experiment_name TEST ...
"""

import sys
sys.path.insert(0, '.')

def test_complete_flow():
    """Test the complete flow from args to dataset"""
    print("="*70)
    print("COMPLETE FLOW TEST: RandomSpeed with NO extra arguments")
    print("="*70)
    
    from train import get_default_args
    from datasets.czech_slr_dataset import CzechSLRDataset
    
    # Step 1: Parse default arguments
    print("\n[Step 1] Parsing default arguments (no user input)...")
    parser = get_default_args()
    args = parser.parse_args([])
    
    print(f"  ✓ use_random_speed: {args.use_random_speed}")
    print(f"  ✓ speed_aug_prob: {args.speed_aug_prob}")
    print(f"  ✓ speed_range: {args.speed_range}")
    
    assert args.use_random_speed == True, "❌ FAIL: use_random_speed should be True!"
    assert args.speed_aug_prob == 0.5, "❌ FAIL: speed_aug_prob should be 0.5!"
    assert args.speed_range == "0.8,1.2", "❌ FAIL: speed_range should be '0.8,1.2'!"
    
    # Step 2: Parse speed_range (as done in train.py)
    print("\n[Step 2] Parsing speed_range string to tuple...")
    speed_range = tuple(map(float, args.speed_range.split(',')))
    print(f"  ✓ Parsed speed_range: {speed_range}")
    assert speed_range == (0.8, 1.2), "❌ FAIL: Parsed speed_range incorrect!"
    
    # Step 3: Dataset initialization (simulating train.py)
    print("\n[Step 3] Initializing dataset with default values...")
    print("  (This simulates: CzechSLRDataset(path, augmentations=True, ")
    print("                    use_random_speed=args.use_random_speed, ...))")
    
    # We can't actually load a real dataset without data files,
    # but we can verify the default parameters
    import inspect
    sig = inspect.signature(CzechSLRDataset.__init__)
    defaults = {
        k: v.default 
        for k, v in sig.parameters.items() 
        if v.default is not inspect.Parameter.empty
    }
    
    print(f"  ✓ Dataset default: use_random_speed={defaults.get('use_random_speed')}")
    print(f"  ✓ Dataset default: speed_aug_prob={defaults.get('speed_aug_prob')}")
    print(f"  ✓ Dataset default: speed_range={defaults.get('speed_range')}")
    
    assert defaults['use_random_speed'] == True, "❌ Dataset default use_random_speed wrong!"
    assert defaults['speed_aug_prob'] == 0.5, "❌ Dataset default speed_aug_prob wrong!"
    assert defaults['speed_range'] == (0.8, 1.2), "❌ Dataset default speed_range wrong!"
    
    # Step 4: Verify augmentation function exists and works
    print("\n[Step 4] Verifying augmentation function...")
    from augmentations.temporal_augmentation import augment_random_speed_fast
    
    # Use more frames to see clear difference
    test_sign = {
        "joint1": [(float(i), float(i)) for i in range(10)],
        "joint2": [(float(i)+0.5, float(i)+0.5) for i in range(10)],
    }
    
    result_slow = augment_random_speed_fast(test_sign, speed_range=(0.8, 0.8), prob=1.0)
    result_fast = augment_random_speed_fast(test_sign, speed_range=(1.2, 1.2), prob=1.0)
    
    print(f"  ✓ Original frames: {len(test_sign['joint1'])}")
    print(f"  ✓ After 0.8x speed (slower): {len(result_slow['joint1'])} frames")
    print(f"  ✓ After 1.2x speed (faster): {len(result_fast['joint1'])} frames")
    
    # Slower speed (0.8x) should have MORE frames, faster speed (1.2x) should have FEWER frames
    assert len(result_slow['joint1']) > len(test_sign['joint1']), "❌ Slower speed should have more frames!"
    assert len(result_fast['joint1']) < len(test_sign['joint1']), "❌ Faster speed should have fewer frames!"
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📋 SUMMARY:")
    print("When you run training with ONLY required arguments:")
    print("  python train.py \\")
    print("      --experiment_name WLASL100 \\")
    print("      --training_set_path datasets/.../train.csv \\")
    print("      --validation_set_path datasets/.../val.csv \\")
    print("      --validation_set from-file \\")
    print("      --num_classes 100")
    print("\nRandomSpeed augmentation WILL BE ENABLED with:")
    print("  • Speed range: 0.8x - 1.2x (80% to 120% speed)")
    print("  • Probability: 50% per training sample")
    print("  • Effect: Simulates different signing speeds via time warping")
    print("  • Implementation: Fast linear interpolation")
    print("\n✅ NO additional arguments needed!")
    print("="*70 + "\n")


def test_command_simulation():
    """Simulate the exact command line that user would run"""
    print("\n" + "="*70)
    print("COMMAND SIMULATION TEST")
    print("="*70)
    print("\nSimulating command:")
    print("  python train.py \\")
    print("      --experiment_name WLASL100 \\")
    print("      --training_set_path ... \\")
    print("      --validation_set_path ... \\")
    print("      --validation_set from-file \\")
    print("      --num_classes 100")
    print("\nParsing...")
    
    from train import get_default_args
    
    parser = get_default_args()
    
    # Simulate minimal command with only required args
    minimal_args = [
        "--experiment_name", "WLASL100",
        "--training_set_path", "dummy/train.csv",
        "--validation_set_path", "dummy/val.csv",
        "--validation_set", "from-file",
        "--num_classes", "100"
    ]
    
    args = parser.parse_args(minimal_args)
    
    print("\n✓ Parsed arguments:")
    print(f"  experiment_name: {args.experiment_name}")
    print(f"  num_classes: {args.num_classes}")
    print(f"  use_random_speed: {args.use_random_speed} (DEFAULT)")
    print(f"  speed_aug_prob: {args.speed_aug_prob} (DEFAULT)")
    print(f"  speed_range: {args.speed_range} (DEFAULT)")
    
    assert args.use_random_speed == True, "❌ FAIL!"
    assert args.speed_aug_prob == 0.5, "❌ FAIL!"
    assert args.speed_range == "0.8,1.2", "❌ FAIL!"
    
    print("\n✅ With only required arguments, RandomSpeed is ENABLED by default!")
    print("="*70)


if __name__ == "__main__":
    test_complete_flow()
    test_command_simulation()
    
    print("\n" + "🎉"*35)
    print("ALL VERIFICATION TESTS PASSED!")
    print("RandomSpeed augmentation works by default!")
    print("🎉"*35 + "\n")
