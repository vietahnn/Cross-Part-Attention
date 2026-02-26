"""
Test script to verify pad_collate_fn works correctly with RandomSpeed augmentation.

This test verifies:
1. Samples with different sequence lengths can be batched together
2. Padding is applied correctly
3. DataLoader with collate_fn works without errors
"""

import sys
sys.path.insert(0, '.')

import torch
from torch.utils.data import DataLoader
from datasets.czech_slr_dataset import pad_collate_fn


def test_pad_collate_fn():
    """Test pad_collate_fn with variable length sequences"""
    print("="*70)
    print("TEST: pad_collate_fn with Variable Sequence Lengths")
    print("="*70)
    
    # Create mock batch with different sequence lengths
    # Simulating what RandomSpeed augmentation might produce
    batch = [
        # Sample 1: 150 frames
        (
            torch.randn(150, 21, 2),  # l_hand
            torch.randn(150, 21, 2),  # r_hand
            torch.randn(150, 12, 2),  # body
            torch.tensor([5])         # label
        ),
        # Sample 2: 180 frames
        (
            torch.randn(180, 21, 2),
            torch.randn(180, 21, 2),
            torch.randn(180, 12, 2),
            torch.tensor([10])
        ),
        # Sample 3: 200 frames
        (
            torch.randn(200, 21, 2),
            torch.randn(200, 21, 2),
            torch.randn(200, 12, 2),
            torch.tensor([3])
        ),
        # Sample 4: 165 frames
        (
            torch.randn(165, 21, 2),
            torch.randn(165, 21, 2),
            torch.randn(165, 12, 2),
            torch.tensor([42])
        ),
    ]
    
    print(f"\n✓ Created mock batch with variable sequence lengths:")
    print(f"  Sample 1: 150 frames")
    print(f"  Sample 2: 180 frames")
    print(f"  Sample 3: 200 frames")
    print(f"  Sample 4: 165 frames")
    print(f"  Expected padded length: 200 frames (max)")
    
    # Apply pad_collate_fn
    l_hands, r_hands, bodies, labels = pad_collate_fn(batch)
    
    print(f"\n✓ After padding:")
    print(f"  l_hands shape: {l_hands.shape}")
    print(f"  r_hands shape: {r_hands.shape}")
    print(f"  bodies shape: {bodies.shape}")
    print(f"  labels shape: {labels.shape}")
    
    # Verify shapes
    batch_size = 4
    max_len = 200
    assert l_hands.shape == (batch_size, max_len, 21, 2), f"❌ l_hands shape mismatch!"
    assert r_hands.shape == (batch_size, max_len, 21, 2), f"❌ r_hands shape mismatch!"
    assert bodies.shape == (batch_size, max_len, 12, 2), f"❌ bodies shape mismatch!"
    assert labels.shape == (batch_size, 1), f"❌ labels shape mismatch!"
    
    print(f"\n✓ All shapes correct!")
    
    # Verify padding (should be zeros)
    # Sample 1 was 150 frames, so frames 150-199 should be zeros
    padding_region = l_hands[0, 150:, :, :]
    assert torch.all(padding_region == 0), "❌ Padding region should be all zeros!"
    
    print(f"✓ Padding verification: frames 150-199 of sample 1 are zeros")
    
    # Verify non-padding region is NOT all zeros
    non_padding_region = l_hands[0, :150, :, :]
    assert not torch.all(non_padding_region == 0), "❌ Non-padding region should not be all zeros!"
    
    print(f"✓ Non-padding verification: frames 0-149 of sample 1 are not zeros")
    
    print("\n✅ pad_collate_fn test PASSED!\n")


def test_with_dataloader():
    """Test that DataLoader works with pad_collate_fn"""
    print("="*70)
    print("TEST: DataLoader with pad_collate_fn")
    print("="*70)
    
    # Create a simple mock dataset
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, lengths):
            self.lengths = lengths
            
        def __len__(self):
            return len(self.lengths)
        
        def __getitem__(self, idx):
            seq_len = self.lengths[idx]
            return (
                torch.randn(seq_len, 21, 2),
                torch.randn(seq_len, 21, 2),
                torch.randn(seq_len, 12, 2),
                torch.tensor([idx])
            )
    
    # Create dataset with variable lengths
    lengths = [150, 180, 200, 165, 170, 190, 160, 175]
    dataset = MockDataset(lengths)
    
    print(f"\n✓ Created mock dataset with {len(lengths)} samples")
    print(f"  Sequence lengths: {lengths}")
    
    # Create DataLoader with pad_collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=pad_collate_fn
    )
    
    print(f"✓ Created DataLoader with batch_size=4 and pad_collate_fn")
    
    # Iterate through batches
    for i, (l_hands, r_hands, bodies, labels) in enumerate(dataloader):
        print(f"\n  Batch {i+1}:")
        print(f"    l_hands: {l_hands.shape}")
        print(f"    r_hands: {r_hands.shape}")
        print(f"    bodies: {bodies.shape}")
        print(f"    labels: {labels.shape}")
        
        # Verify all samples in batch have same length
        assert l_hands.shape[0] == 4 or (i == 1 and l_hands.shape[0] == 0), "❌ Batch size mismatch!"
        assert l_hands.shape[1] == r_hands.shape[1] == bodies.shape[1], "❌ Sequence lengths mismatch!"
    
    print("\n✅ DataLoader test PASSED!\n")


def test_extreme_cases():
    """Test edge cases"""
    print("="*70)
    print("TEST: Edge Cases")
    print("="*70)
    
    # Test 1: All same length (no padding needed)
    print("\n  Test 1: All same length (no padding)")
    batch_same = [
        (torch.randn(100, 21, 2), torch.randn(100, 21, 2), torch.randn(100, 12, 2), torch.tensor([0])),
        (torch.randn(100, 21, 2), torch.randn(100, 21, 2), torch.randn(100, 12, 2), torch.tensor([1])),
    ]
    l_hands, r_hands, bodies, labels = pad_collate_fn(batch_same)
    assert l_hands.shape == (2, 100, 21, 2), "❌ Same-length test failed!"
    print("    ✓ No padding needed, shapes correct")
    
    # Test 2: Very short sequence
    print("\n  Test 2: Very short sequence (50 frames)")
    batch_short = [
        (torch.randn(50, 21, 2), torch.randn(50, 21, 2), torch.randn(50, 12, 2), torch.tensor([0])),
        (torch.randn(200, 21, 2), torch.randn(200, 21, 2), torch.randn(200, 12, 2), torch.tensor([1])),
    ]
    l_hands, r_hands, bodies, labels = pad_collate_fn(batch_short)
    assert l_hands.shape == (2, 200, 21, 2), "❌ Short sequence test failed!"
    # Verify 150 frames were padded (50 -> 200)
    assert torch.all(l_hands[0, 50:, :, :] == 0), "❌ Padding not correct!"
    print("    ✓ 150 frames padded correctly")
    
    # Test 3: Single sample batch
    print("\n  Test 3: Single sample batch")
    batch_single = [
        (torch.randn(100, 21, 2), torch.randn(100, 21, 2), torch.randn(100, 12, 2), torch.tensor([0]))
    ]
    l_hands, r_hands, bodies, labels = pad_collate_fn(batch_single)
    assert l_hands.shape == (1, 100, 21, 2), "❌ Single sample test failed!"
    print("    ✓ Single sample handled correctly")
    
    print("\n✅ All edge cases PASSED!\n")


def main():
    print("\n" + "🎯"*35)
    print("PAD_COLLATE_FN VERIFICATION TEST")
    print("🎯"*35 + "\n")
    
    try:
        # Test 1: Basic padding functionality
        test_pad_collate_fn()
        
        # Test 2: DataLoader integration
        test_with_dataloader()
        
        # Test 3: Edge cases
        test_extreme_cases()
        
        # Summary
        print("\n" + "🎉"*35)
        print("ALL TESTS PASSED!")
        print("🎉"*35 + "\n")
        
        print("📋 SUMMARY:")
        print("="*70)
        print("✅ pad_collate_fn handles variable sequence lengths correctly")
        print("✅ Padding is applied with zeros to match max length in batch")
        print("✅ DataLoader works correctly with custom collate_fn")
        print("✅ Edge cases (same length, very short, single sample) handled")
        print("="*70)
        
        print("\n💡 HOW IT WORKS:")
        print("  1. RandomSpeed changes sequence lengths (e.g., 200 → 186 frames)")
        print("  2. pad_collate_fn finds max length in batch (e.g., max=200)")
        print("  3. Shorter sequences are padded with zeros to match max length")
        print("  4. All sequences in batch have same length → can be stacked")
        print("  5. PyTorch DataLoader can create batches successfully")
        
        print("\n🔥 FIX APPLIED:")
        print("  • Added pad_collate_fn to datasets/czech_slr_dataset.py")
        print("  • Updated all DataLoader instances in train.py")
        print("  • RandomSpeed augmentation now works without errors!")
        
        print("\n🚀 READY TO TRAIN WITH RANDOMSPEED!\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
