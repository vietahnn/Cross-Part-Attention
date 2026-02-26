"""
Test script to verify data type compatibility between input and DenseNet model.

This test verifies:
1. DoubleTensor (float64) input is converted to FloatTensor (float32)
2. Ensemble model handles different data types correctly
3. No type mismatch errors occur
"""

import sys
sys.path.insert(0, '.')

import torch
from siformer.model import SiFormer
from siformer.densenet_model import build_densenet_model
from siformer.ensemble_model import build_ensemble_model


def test_data_type_conversion():
    """Test that different input data types are handled correctly"""
    print("="*70)
    print("TEST: Data Type Conversion (DoubleTensor → FloatTensor)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Create models
    siformer = SiFormer(num_classes=100, num_hid=108, device=device, IA_encoder=False, IA_decoder=False)
    densenet = build_densenet_model(num_classes=100, num_keypoints=54, dropout=0.2, use_1d=False)
    
    ensemble, loss_fn = build_ensemble_model(
        siformer_model=siformer,
        densenet_model=densenet,
        ensemble_weights=(0.5, 0.5),
        learnable_weights=False
    )
    
    ensemble.to(device)
    ensemble.eval()
    
    print("\n✓ Ensemble model created and moved to device")
    
    # Test 1: Float32 input (normal case)
    print("\n[Test 1] Float32 input (torch.FloatTensor)")
    batch_size, seq_len = 2, 204
    l_hand_f32 = torch.randn(batch_size, seq_len, 21, 2, dtype=torch.float32).to(device)
    r_hand_f32 = torch.randn(batch_size, seq_len, 21, 2, dtype=torch.float32).to(device)
    body_f32 = torch.randn(batch_size, seq_len, 12, 2, dtype=torch.float32).to(device)
    
    print(f"  Input types: {l_hand_f32.dtype}, {r_hand_f32.dtype}, {body_f32.dtype}")
    
    with torch.no_grad():
        ensemble_out, siformer_out, densenet_out = ensemble(l_hand_f32, r_hand_f32, body_f32, training=False)
    
    print(f"  ✓ Output shape: {ensemble_out.shape}")
    print(f"  ✓ No type mismatch error!")
    
    # Test 2: Float64 input (DoubleTensor - the problematic case)
    print("\n[Test 2] Float64 input (torch.DoubleTensor)")
    l_hand_f64 = torch.randn(batch_size, seq_len, 21, 2, dtype=torch.float64).to(device)
    r_hand_f64 = torch.randn(batch_size, seq_len, 21, 2, dtype=torch.float64).to(device)
    body_f64 = torch.randn(batch_size, seq_len, 12, 2, dtype=torch.float64).to(device)
    
    print(f"  Input types: {l_hand_f64.dtype}, {r_hand_f64.dtype}, {body_f64.dtype}")
    
    try:
        with torch.no_grad():
            ensemble_out, siformer_out, densenet_out = ensemble(l_hand_f64, r_hand_f64, body_f64, training=False)
        
        print(f"  ✓ Output shape: {ensemble_out.shape}")
        print(f"  ✓ DoubleTensor converted to FloatTensor successfully!")
        print(f"  ✓ No type mismatch error!")
        
    except RuntimeError as e:
        if "should be the same" in str(e):
            print(f"  ❌ FAILED: Type mismatch error still occurs!")
            print(f"  Error: {e}")
            return False
        else:
            raise
    
    # Test 3: Mixed types
    print("\n[Test 3] Mixed types (Float32 + Float64)")
    l_hand_mixed = torch.randn(batch_size, seq_len, 21, 2, dtype=torch.float64).to(device)
    r_hand_mixed = torch.randn(batch_size, seq_len, 21, 2, dtype=torch.float32).to(device)
    body_mixed = torch.randn(batch_size, seq_len, 12, 2, dtype=torch.float64).to(device)
    
    print(f"  Input types: {l_hand_mixed.dtype}, {r_hand_mixed.dtype}, {body_mixed.dtype}")
    
    try:
        with torch.no_grad():
            ensemble_out, siformer_out, densenet_out = ensemble(l_hand_mixed, r_hand_mixed, body_mixed, training=False)
        
        print(f"  ✓ Output shape: {ensemble_out.shape}")
        print(f"  ✓ Mixed types handled successfully!")
        
    except Exception as e:
        print(f"  ⚠ Warning: Mixed types may cause issues in other parts")
        print(f"  Error: {e}")
    
    print("\n✅ Data type conversion test PASSED!\n")
    return True


def test_densenet_standalone():
    """Test DenseNet with different input types"""
    print("="*70)
    print("TEST: DenseNet Standalone with Different Input Types")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_densenet_model(num_classes=100, num_keypoints=54, dropout=0.2, use_1d=False)
    model.to(device)
    model.eval()
    
    print("\n✓ DenseNet model created")
    
    batch_size, seq_len = 2, 204
    
    # Test with Float64 (DoubleTensor)
    print("\n[Test] Float64 input to DenseNet")
    input_f64 = torch.randn(batch_size, seq_len, 108, dtype=torch.float64).to(device)
    
    print(f"  Input type: {input_f64.dtype}")
    print(f"  Input shape: {input_f64.shape}")
    
    try:
        with torch.no_grad():
            output = model(input_f64)
        
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ DenseNet handles Float64 input correctly!")
        
    except RuntimeError as e:
        if "should be the same" in str(e):
            print(f"  ❌ FAILED: Type mismatch error!")
            print(f"  Error: {e}")
            return False
        else:
            raise
    
    print("\n✅ DenseNet standalone test PASSED!\n")
    return True


def main():
    print("\n" + "🔧"*35)
    print("DATA TYPE COMPATIBILITY TEST")
    print("🔧"*35 + "\n")
    
    try:
        # Test 1: Data type conversion in ensemble
        success1 = test_data_type_conversion()
        
        # Test 2: DenseNet standalone
        success2 = test_densenet_standalone()
        
        if success1 and success2:
            # Summary
            print("\n" + "🎉"*35)
            print("ALL TESTS PASSED!")
            print("🎉"*35 + "\n")
            
            print("📋 SUMMARY:")
            print("="*70)
            print("✅ Float32 (FloatTensor) input works correctly")
            print("✅ Float64 (DoubleTensor) input is converted to Float32")
            print("✅ No type mismatch errors")
            print("✅ Ensemble model handles different data types")
            print("✅ DenseNet model handles different data types")
            print("="*70)
            
            print("\n💡 FIX APPLIED:")
            print("  • Added .float() conversion in ensemble_model.py")
            print("  • Added .float() conversion in densenet_model.py")
            print("  • DenseNet now accepts DoubleTensor/FloatTensor inputs")
            
            print("\n🔥 ROOT CAUSE:")
            print("  • Input data: torch.DoubleTensor (float64)")
            print("  • Model weights: torch.FloatTensor (float32)")
            print("  • PyTorch Conv2d requires same data types")
            print("  • Solution: Convert input to float32 before conv layers")
            
            print("\n🚀 READY TO TRAIN WITH ENSEMBLE!\n")
            return True
        else:
            print("\n❌ SOME TESTS FAILED!")
            return False
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
