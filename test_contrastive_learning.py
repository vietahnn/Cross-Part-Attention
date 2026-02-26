"""
Test script for Contrastive Learning Implementation

This script validates that supervised contrastive loss and center loss work correctly
with the Siformer training pipeline.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from siformer.contrastive_loss import (
    SupervisedContrastiveLoss,
    CenterLoss,
    PrototypeLoss,
    HybridContrastiveCenterLoss,
    compute_feature_statistics
)


def test_supervised_contrastive_loss():
    """Test supervised contrastive loss"""
    print("\n" + "="*70)
    print("TEST 1: Supervised Contrastive Loss")
    print("="*70)
    
    batch_size = 64
    feature_dim = 256
    num_classes = 100
    
    # Create features with some structure (same class = similar features)
    features = []
    labels = []
    for class_id in range(10):  # 10 classes
        for _ in range(6):  # 6 samples per class
            # Base feature for this class + noise
            base = torch.randn(feature_dim) + class_id * 2.0
            noise = torch.randn(feature_dim) * 0.5
            features.append(base + noise)
            labels.append(class_id)
    
    features = torch.stack(features[:batch_size])
    labels = torch.tensor(labels[:batch_size])
    
    print(f"Batch size: {batch_size}")
    print(f"Feature dim: {feature_dim}")
    print(f"Unique classes: {labels.unique().size(0)}")
    
    # Test with different temperatures
    for temp in [0.05, 0.07, 0.1]:
        loss_fn = SupervisedContrastiveLoss(temperature=temp)
        loss = loss_fn(features, labels)
        print(f"  - Temperature {temp:.2f}: Loss = {loss.item():.4f}")
    
    print("✓ Supervised Contrastive Loss passed!")


def test_center_loss():
    """Test center loss"""
    print("\n" + "="*70)
    print("TEST 2: Center Loss")
    print("="*70)
    
    batch_size = 32
    feature_dim = 128
    num_classes = 100
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test center loss
    center_loss = CenterLoss(num_classes, feature_dim, lambda_center=0.003, use_cuda=False)
    loss = center_loss(features, labels)
    
    print(f"Initial center loss: {loss.item():.6f}")
    
    # Update centers
    center_loss.update_centers(features, labels, alpha=0.5)
    loss_after = center_loss(features, labels)
    
    print(f"After update: {loss_after.item():.6f}")
    print(f"Centers shape: {center_loss.get_centers().shape}")
    
    assert loss_after.item() < loss.item(), "Center loss should decrease after update!"
    
    print("✓ Center Loss passed!")


def test_prototype_loss():
    """Test prototype loss"""
    print("\n" + "="*70)
    print("TEST 3: Prototype Loss")
    print("="*70)
    
    batch_size = 48
    feature_dim = 128
    
    features = torch.randn(batch_size, feature_dim)
    # Create labels with multiple samples per class
    labels = torch.tensor([i // 4 for i in range(batch_size)])  # 4 samples per class
    
    proto_loss = PrototypeLoss(temperature=0.1)
    loss = proto_loss(features, labels)
    
    print(f"Prototype loss: {loss.item():.6f}")
    print(f"Num classes in batch: {labels.unique().size(0)}")
    
    print("✓ Prototype Loss passed!")


def test_hybrid_loss():
    """Test hybrid contrastive + center loss"""
    print("\n" + "="*70)
    print("TEST 4: Hybrid Contrastive + Center Loss")
    print("="*70)
    
    batch_size = 32
    feature_dim = 256
    num_classes = 100
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test with different weight combinations
    configs = [
        {'beta': 0.5, 'gamma': 0.1, 'name': 'Balanced'},
        {'beta': 1.0, 'gamma': 0.0, 'name': 'Contrastive only'},
        {'beta': 0.0, 'gamma': 1.0, 'name': 'Center only'},
        {'beta': 0.8, 'gamma': 0.2, 'name': 'Heavy contrastive'}
    ]
    
    for config in configs:
        hybrid = HybridContrastiveCenterLoss(
            num_classes, feature_dim,
            beta=config['beta'],
            gamma=config['gamma'],
            use_cuda=False
        )
        losses = hybrid(features, labels)
        print(f"{config['name']:20s} - Total: {losses['total'].item():.4f}, "
              f"Con: {losses['contrastive'].item():.4f}, "
              f"Center: {losses['center'].item():.6f}")
    
    print("✓ Hybrid Loss passed!")


def test_gradient_flow():
    """Test that gradients flow through losses"""
    print("\n" + "="*70)
    print("TEST 5: Gradient Flow")
    print("="*70)
    
    batch_size = 16
    feature_dim = 128
    num_classes = 50
    
    features = torch.randn(batch_size, feature_dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test contrastive loss gradient
    supcon = SupervisedContrastiveLoss()
    loss_con = supcon(features, labels)
    loss_con.backward()
    
    print(f"Contrastive loss gradient norm: {features.grad.norm().item():.6f}")
    assert features.grad is not None, "No gradient!"
    
    # Reset gradient
    features.grad.zero_()
    
    # Test center loss gradient
    center_loss = CenterLoss(num_classes, feature_dim, use_cuda=False)
    loss_center = center_loss(features, labels)
    loss_center.backward()
    
    print(f"Center loss gradient norm: {features.grad.norm().item():.6f}")
    assert features.grad is not None, "No gradient!"
    
    print("✓ Gradient Flow passed!")


def test_feature_statistics():
    """Test feature statistics computation"""
    print("\n" + "="*70)
    print("TEST 6: Feature Statistics")
    print("="*70)
    
    # Create structured features (tight clusters)
    features = []
    labels = []
    for class_id in range(5):
        for _ in range(10):
            base = torch.randn(128) + class_id * 5.0  # Well-separated classes
            noise = torch.randn(128) * 0.1  # Low intra-class variance
            features.append(base + noise)
            labels.append(class_id)
    
    features = torch.stack(features)
    labels = torch.tensor(labels)
    
    stats = compute_feature_statistics(features, labels)
    
    print(f"Intra-class variance: {stats['intra_class_variance']:.4f} (should be low)")
    print(f"Inter-class distance: {stats['inter_class_distance']:.4f} (should be high)")
    print(f"Feature norm: {stats['feature_norm']:.4f}")
    
    # Check that inter-class distance > intra-class variance (good separation)
    assert stats['inter_class_distance'] > stats['intra_class_variance'], \
        "Inter-class distance should be larger than intra-class variance!"
    
    print("✓ Feature Statistics passed!")


def test_cuda_compatibility():
    """Test CUDA compatibility if available"""
    print("\n" + "="*70)
    print("TEST 7: CUDA Compatibility")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return
    
    device = torch.device("cuda")
    batch_size = 16
    feature_dim = 128
    num_classes = 50
    
    features = torch.randn(batch_size, feature_dim).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Test hybrid loss on CUDA
    hybrid = HybridContrastiveCenterLoss(num_classes, feature_dim, use_cuda=True)
    hybrid.to(device)
    
    losses = hybrid(features, labels)
    
    print(f"CUDA device: {device}")
    print(f"Features device: {features.device}")
    print(f"Loss device: {losses['total'].device}")
    print(f"Loss value: {losses['total'].item():.4f}")
    
    assert losses['total'].device.type == 'cuda', "Loss should be on CUDA!"
    
    print("✓ CUDA Compatibility passed!")


def test_batch_variations():
    """Test with different batch compositions"""
    print("\n" + "="*70)
    print("TEST 8: Batch Variations")
    print("="*70)
    
    feature_dim = 128
    num_classes = 100
    
    # Test 1: Small batch
    features_small = torch.randn(4, feature_dim)
    labels_small = torch.randint(0, num_classes, (4,))
    
    hybrid = HybridContrastiveCenterLoss(num_classes, feature_dim, use_cuda=False)
    loss_small = hybrid(features_small, labels_small)
    print(f"Small batch (4): {loss_small['total'].item():.4f}")
    
    # Test 2: All same class
    features_same = torch.randn(16, feature_dim)
    labels_same = torch.zeros(16, dtype=torch.long)
    
    loss_same = hybrid(features_same, labels_same)
    print(f"All same class: {loss_same['total'].item():.4f}")
    
    # Test 3: All different classes
    features_diff = torch.randn(16, feature_dim)
    labels_diff = torch.arange(16)
    
    loss_diff = hybrid(features_diff, labels_diff)
    print(f"All different classes: {loss_diff['total'].item():.4f}")
    
    print("✓ Batch Variations passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*16 + "CONTRASTIVE LEARNING TEST SUITE" + " "*21 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        test_supervised_contrastive_loss()
        test_center_loss()
        test_prototype_loss()
        test_hybrid_loss()
        test_gradient_flow()
        test_feature_statistics()
        test_cuda_compatibility()
        test_batch_variations()
        
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + " "*20 + "ALL TESTS PASSED! ✓✓✓" + " "*27 + "█")
        print("█" + " "*68 + "█")
        print("█"*70 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
