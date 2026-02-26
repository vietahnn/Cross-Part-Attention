"""
Test script to verify ensemble model (Siformer + DenseNet) setup.

This script verifies:
1. DenseNet model can be built
2. Ensemble model can be built
3. Forward pass works correctly
4. Loss computation works
5. Default parameters are set correctly
"""

import sys
sys.path.insert(0, '.')

import torch
from siformer.model import SiFormer
from siformer.densenet_model import build_densenet_model
from siformer.ensemble_model import build_ensemble_model


def test_densenet_model():
    """Test DenseNet model creation and forward pass"""
    print("="*70)
    print("TEST 1: DenseNet Model")
    print("="*70)
    
    model = build_densenet_model(
        num_classes=100,
        num_keypoints=45,
        dropout=0.2,
        pretrained=False,
        use_1d=False
    )
    
    # Test forward pass
    batch_size, seq_len, features = 4, 50, 90  # 45 keypoints * 2 coords = 90
    test_input = torch.randn(batch_size, seq_len, features)
    
    output = model(test_input)
    
    print(f"✓ Input shape: {test_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: ({batch_size}, 100)")
    
    assert output.shape == (batch_size, 100), f"❌ Output shape mismatch!"
    print("✅ DenseNet model test PASSED!\n")
    
    return model


def test_ensemble_model():
    """Test ensemble model creation and forward pass"""
    print("="*70)
    print("TEST 2: Ensemble Model (Siformer + DenseNet)")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Create Siformer (disable IA_encoder for simpler testing)
    siformer = SiFormer(num_classes=100, num_hid=108, device=device, IA_encoder=False, IA_decoder=False)
    
    # Create DenseNet
    densenet = build_densenet_model(
        num_classes=100,
        num_keypoints=54,  # 21 (left_hand) + 21 (right_hand) + 12 (body) = 54
        dropout=0.2,
        use_1d=False
    )
    
    # Build ensemble
    ensemble, loss_fn = build_ensemble_model(
        siformer_model=siformer,
        densenet_model=densenet,
        ensemble_weights=(0.5, 0.5),
        ensemble_method='weighted_average',
        learnable_weights=False,
        aux_loss_weight=0.3
    )
    
    print("\n✓ Ensemble model created successfully")
    
    # Test forward pass
    batch_size, seq_len = 4, 204  # SiFormer expects seq_len=204 by default
    l_hand = torch.randn(batch_size, seq_len, 21, 2)  # 21 keypoints, 2 coords
    r_hand = torch.randn(batch_size, seq_len, 21, 2)  # 21 keypoints, 2 coords
    body = torch.randn(batch_size, seq_len, 12, 2)  # 12 keypoints, 2 coords (neck, shoulders, hips, etc)
    
    ensemble_out, siformer_out, densenet_out = ensemble(l_hand, r_hand, body, training=False)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Ensemble output: {ensemble_out.shape}")
    print(f"  Siformer output: {siformer_out.shape}")
    print(f"  DenseNet output: {densenet_out.shape}")
    
    # Test loss
    targets = torch.randint(0, 100, (batch_size,))
    total_loss, loss_dict = loss_fn(ensemble_out, siformer_out, densenet_out, targets)
    
    print(f"\n✓ Loss computation successful!")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Ensemble loss: {loss_dict['ensemble']:.4f}")
    print(f"  Siformer loss: {loss_dict['siformer']:.4f}")
    print(f"  DenseNet loss: {loss_dict['densenet']:.4f}")
    
    print("\n✅ Ensemble model test PASSED!\n")
    
    return ensemble, loss_fn


def test_default_arguments():
    """Test that default arguments are set correctly"""
    print("="*70)
    print("TEST 3: Default Arguments")
    print("="*70)
    
    from train import get_default_args
    
    parser = get_default_args()
    args = parser.parse_args([])
    
    print(f"✓ use_ensemble: {args.use_ensemble}")
    print(f"✓ ensemble_weights: {args.ensemble_weights}")
    print(f"✓ ensemble_method: {args.ensemble_method}")
    print(f"✓ learnable_ensemble_weights: {args.learnable_ensemble_weights}")
    print(f"✓ aux_loss_weight: {args.aux_loss_weight}")
    print(f"✓ densenet_dropout: {args.densenet_dropout}")
    print(f"✓ densenet_use_1d: {args.densenet_use_1d}")
    print(f"✓ densenet_pretrained: {args.densenet_pretrained}")
    
    # Verify defaults
    assert args.use_ensemble == True, "❌ use_ensemble should be True by default!"
    assert args.ensemble_weights == "0.5,0.5", "❌ ensemble_weights should be '0.5,0.5' by default!"
    assert args.ensemble_method == "weighted_average", "❌ ensemble_method should be 'weighted_average' by default!"
    assert args.learnable_ensemble_weights == False, "❌ learnable_ensemble_weights should be False by default!"
    assert args.aux_loss_weight == 0.3, "❌ aux_loss_weight should be 0.3 by default!"
    assert args.densenet_dropout == 0.2, "❌ densenet_dropout should be 0.2 by default!"
    assert args.densenet_use_1d == False, "❌ densenet_use_1d should be False by default!"
    assert args.densenet_pretrained == False, "❌ densenet_pretrained should be False by default!"
    
    print("\n✅ Default arguments test PASSED!\n")


def test_command_simulation():
    """Simulate minimal training command"""
    print("="*70)
    print("TEST 4: Command Simulation")
    print("="*70)
    
    from train import get_default_args
    
    parser = get_default_args()
    
    # Simulate minimal command
    minimal_args = [
        "--experiment_name", "WLASL100_ensemble",
        "--training_set_path", "datasets/WLASL100/WLASL100_train_25fps.csv",
        "--validation_set_path", "datasets/WLASL100/WLASL100_val_25fps.csv",
        "--validation_set", "from-file",
        "--num_classes", "100"
    ]
    
    args = parser.parse_args(minimal_args)
    
    print("\n✓ Command parsed successfully:")
    print(f"  experiment_name: {args.experiment_name}")
    print(f"  num_classes: {args.num_classes}")
    print(f"  use_ensemble: {args.use_ensemble} (DEFAULT)")
    print(f"  ensemble_weights: {args.ensemble_weights} (DEFAULT)")
    print(f"  ensemble_method: {args.ensemble_method} (DEFAULT)")
    
    assert args.use_ensemble == True, "❌ Ensemble should be enabled by default!"
    
    print("\n✅ With minimal arguments, ENSEMBLE is ENABLED by default!")
    print("\n✅ Command simulation test PASSED!\n")


def main():
    print("\n" + "🎯"*35)
    print("ENSEMBLE MODEL VERIFICATION TEST")
    print("🎯"*35 + "\n")
    
    try:
        # Test 1: DenseNet model
        test_densenet_model()
        
        # Test 2: Ensemble model
        test_ensemble_model()
        
        # Test 3: Default arguments
        test_default_arguments()
        
        # Test 4: Command simulation
        test_command_simulation()
        
        # Summary
        print("\n" + "🎉"*35)
        print("ALL TESTS PASSED!")
        print("🎉"*35 + "\n")
        
        print("📋 SUMMARY:")
        print("="*70)
        print("✅ DenseNet model works correctly")
        print("✅ Ensemble model (Siformer + DenseNet) works correctly")
        print("✅ All default parameters are set correctly")
        print("✅ Ensemble is ENABLED by default")
        print("="*70)
        
        print("\n💡 TO USE ENSEMBLE:")
        print("Simply run the training command WITHOUT any extra arguments:")
        print("\n  python train.py \\")
        print("      --experiment_name WLASL100_ensemble \\")
        print("      --training_set_path datasets/WLASL100/WLASL100_train_25fps.csv \\")
        print("      --validation_set_path datasets/WLASL100/WLASL100_val_25fps.csv \\")
        print("      --validation_set from-file \\")
        print("      --num_classes 100")
        
        print("\n🔥 Ensemble will combine:")
        print("  • Siformer (Transformer-based): 50% weight")
        print("  • DenseNet (CNN-based): 50% weight")
        print("  • Method: Weighted average")
        print("  • Auxiliary loss weight: 0.3")
        
        print("\n🚀 READY TO TRAIN WITH ENSEMBLE!\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
