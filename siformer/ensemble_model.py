"""
Ensemble model combining Siformer (Transformer) and DenseNet for sign language recognition.

Ensemble strategies:
1. Average: Simple average of predictions
2. Weighted Average: Weighted combination based on validation performance
3. Learned Ensemble: Trainable weights (optional)

This follows the approach from SL-TSSI-DenseNet project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    """
    Ensemble of Siformer and DenseNet models.
    
    Combines predictions using weighted average or other ensemble strategies.
    """
    def __init__(self, siformer_model, densenet_model, 
                 ensemble_weights=(0.5, 0.5), 
                 ensemble_method='weighted_average',
                 learnable_weights=False):
        """
        Args:
            siformer_model: Trained or training Siformer model
            densenet_model: Trained or training DenseNet model
            ensemble_weights: Tuple of (weight_siformer, weight_densenet), must sum to 1.0
            ensemble_method: 'weighted_average', 'average', 'max', or 'learned'
            learnable_weights: If True, make ensemble weights trainable
        """
        super(EnsembleModel, self).__init__()
        
        self.siformer = siformer_model
        self.densenet = densenet_model
        self.ensemble_method = ensemble_method
        
        # Validate weights
        if ensemble_method == 'weighted_average':
            assert len(ensemble_weights) == 2, "ensemble_weights must be (w1, w2)"
            assert abs(sum(ensemble_weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            
            if learnable_weights:
                # Make weights learnable parameters
                self.weight_siformer = nn.Parameter(torch.tensor(ensemble_weights[0]))
                self.weight_densenet = nn.Parameter(torch.tensor(ensemble_weights[1]))
                print(f"✓ Ensemble with LEARNABLE weights: Siformer={ensemble_weights[0]:.2f}, DenseNet={ensemble_weights[1]:.2f}")
            else:
                # Fixed weights
                self.register_buffer('weight_siformer', torch.tensor(ensemble_weights[0]))
                self.register_buffer('weight_densenet', torch.tensor(ensemble_weights[1]))
                print(f"✓ Ensemble with FIXED weights: Siformer={ensemble_weights[0]:.2f}, DenseNet={ensemble_weights[1]:.2f}")
        else:
            self.weight_siformer = None
            self.weight_densenet = None
            print(f"✓ Ensemble method: {ensemble_method}")
        
        self.learnable_weights = learnable_weights
        
    def forward(self, l_hand, r_hand, body, training=True):
        """
        Forward pass through both models and ensemble predictions.
        
        Args:
            l_hand: Left hand features (batch, seq_len, 21, 2)
            r_hand: Right hand features (batch, seq_len, 21, 2)
            body: Body features (batch, seq_len, 3, 2)
            training: Whether in training mode
            
        Returns:
            ensemble_output: Combined predictions from both models
            siformer_output: Siformer predictions (for auxiliary loss)
            densenet_output: DenseNet predictions (for auxiliary loss)
        """
        # Get Siformer predictions
        siformer_output = self.siformer(l_hand, r_hand, body, training=training)
        
        # Prepare input for DenseNet: flatten and concatenate all features
        # Shape: (batch, seq_len, num_keypoints, 2)
        # Flatten to: (batch, seq_len, num_keypoints * 2)
        l_hand_flat = l_hand.view(l_hand.size(0), l_hand.size(1), -1)  # (batch, seq_len, 42)
        r_hand_flat = r_hand.view(r_hand.size(0), r_hand.size(1), -1)  # (batch, seq_len, 42)
        body_flat = body.view(body.size(0), body.size(1), -1)  # (batch, seq_len, 6)
        
        densenet_input = torch.cat([l_hand_flat, r_hand_flat, body_flat], dim=-1)  # (batch, seq_len, 90)
        
        # Convert to float32 to match DenseNet weights (fix DoubleTensor/FloatTensor mismatch)
        densenet_input = densenet_input.float()
        
        # Get DenseNet predictions
        densenet_output = self.densenet(densenet_input)
        
        # Ensemble the predictions
        if self.ensemble_method == 'average':
            # Simple average
            ensemble_output = (siformer_output + densenet_output) / 2.0
            
        elif self.ensemble_method == 'weighted_average':
            # Weighted average
            if self.learnable_weights:
                # Normalize weights to sum to 1 using sigmoid
                w_siformer = torch.sigmoid(self.weight_siformer)
                w_densenet = 1.0 - w_siformer
                ensemble_output = w_siformer * siformer_output + w_densenet * densenet_output
            else:
                # Fixed weights
                ensemble_output = self.weight_siformer * siformer_output + self.weight_densenet * densenet_output
                
        elif self.ensemble_method == 'max':
            # Take maximum probability for each class
            ensemble_output = torch.max(siformer_output, densenet_output)
            
        elif self.ensemble_method == 'learned':
            # Learned combination (not implemented yet)
            raise NotImplementedError("Learned ensemble not yet implemented")
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_output, siformer_output, densenet_output
    
    def get_ensemble_weights(self):
        """Get current ensemble weights"""
        if self.weight_siformer is not None:
            if self.learnable_weights:
                w_siformer = torch.sigmoid(self.weight_siformer).item()
                w_densenet = 1.0 - w_siformer
            else:
                w_siformer = self.weight_siformer.item()
                w_densenet = self.weight_densenet.item()
            return w_siformer, w_densenet
        else:
            return None, None


class EnsembleLoss(nn.Module):
    """
    Combined loss for ensemble training.
    
    Computes loss for:
    1. Ensemble predictions (main loss)
    2. Siformer predictions (auxiliary loss)
    3. DenseNet predictions (auxiliary loss)
    """
    def __init__(self, aux_weight=0.3, loss_fn=None):
        """
        Args:
            aux_weight: Weight for auxiliary losses (default 0.3)
                       Total loss = ensemble_loss + aux_weight * (siformer_loss + densenet_loss)
            loss_fn: Loss function (default: CrossEntropyLoss)
        """
        super(EnsembleLoss, self).__init__()
        self.aux_weight = aux_weight
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
        print(f"✓ EnsembleLoss initialized: aux_weight={aux_weight}")
        
    def forward(self, ensemble_output, siformer_output, densenet_output, targets):
        """
        Compute combined loss.
        
        Args:
            ensemble_output: Ensemble predictions (batch, num_classes)
            siformer_output: Siformer predictions (batch, num_classes)
            densenet_output: DenseNet predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses for logging
        """
        # Main ensemble loss
        ensemble_loss = self.loss_fn(ensemble_output, targets)
        
        # Auxiliary losses
        siformer_loss = self.loss_fn(siformer_output, targets)
        densenet_loss = self.loss_fn(densenet_output, targets)
        
        # Combined loss
        total_loss = ensemble_loss + self.aux_weight * (siformer_loss + densenet_loss)
        
        # Return losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'ensemble': ensemble_loss.item(),
            'siformer': siformer_loss.item(),
            'densenet': densenet_loss.item()
        }
        
        return total_loss, loss_dict


def build_ensemble_model(siformer_model, 
                         densenet_model,
                         ensemble_weights=(0.5, 0.5),
                         ensemble_method='weighted_average',
                         learnable_weights=False,
                         aux_loss_weight=0.3):
    """
    Build ensemble model and loss function.
    
    Args:
        siformer_model: Siformer model instance
        densenet_model: DenseNet model instance
        ensemble_weights: Tuple of (weight_siformer, weight_densenet)
        ensemble_method: 'weighted_average', 'average', or 'max'
        learnable_weights: Make ensemble weights trainable
        aux_loss_weight: Weight for auxiliary losses
        
    Returns:
        ensemble_model: EnsembleModel instance
        ensemble_loss: EnsembleLoss instance
    """
    ensemble_model = EnsembleModel(
        siformer_model=siformer_model,
        densenet_model=densenet_model,
        ensemble_weights=ensemble_weights,
        ensemble_method=ensemble_method,
        learnable_weights=learnable_weights
    )
    
    ensemble_loss = EnsembleLoss(aux_weight=aux_loss_weight)
    
    print("\n" + "="*70)
    print("🔥 ENSEMBLE MODEL BUILT")
    print("="*70)
    print(f"  Siformer: Transformer-based model")
    print(f"  DenseNet: CNN-based model")
    print(f"  Ensemble method: {ensemble_method}")
    if ensemble_method == 'weighted_average':
        print(f"  Weights: Siformer={ensemble_weights[0]:.2f}, DenseNet={ensemble_weights[1]:.2f}")
        print(f"  Learnable weights: {learnable_weights}")
    print(f"  Auxiliary loss weight: {aux_loss_weight}")
    print("="*70 + "\n")
    
    return ensemble_model, ensemble_loss


if __name__ == "__main__":
    # Test ensemble model
    print("Testing EnsembleModel...\n")
    
    from siformer.model import SiFormer
    from siformer.densenet_model import build_densenet_model
    
    # Create mock models
    device = torch.device('cpu')
    siformer = SiFormer(num_classes=100, num_hid=108, device=device)
    densenet = build_densenet_model(num_classes=100, num_keypoints=45, use_1d=False)
    
    # Build ensemble
    ensemble, loss_fn = build_ensemble_model(
        siformer_model=siformer,
        densenet_model=densenet,
        ensemble_weights=(0.5, 0.5),
        learnable_weights=False
    )
    
    # Test forward pass
    batch_size, seq_len = 4, 50
    l_hand = torch.randn(batch_size, seq_len, 42)
    r_hand = torch.randn(batch_size, seq_len, 42)
    body = torch.randn(batch_size, seq_len, 6)
    tgt = torch.zeros(batch_size, 1, 108)
    
    src = [l_hand, r_hand, body]
    
    ensemble_out, siformer_out, densenet_out = ensemble(src, tgt, training=False)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Ensemble output shape: {ensemble_out.shape}")
    print(f"  Siformer output shape: {siformer_out.shape}")
    print(f"  DenseNet output shape: {densenet_out.shape}")
    
    # Test loss
    targets = torch.randint(0, 100, (batch_size,))
    total_loss, loss_dict = loss_fn(ensemble_out, siformer_out, densenet_out, targets)
    
    print(f"\n✓ Loss computation successful!")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Ensemble loss: {loss_dict['ensemble']:.4f}")
    print(f"  Siformer loss: {loss_dict['siformer']:.4f}")
    print(f"  DenseNet loss: {loss_dict['densenet']:.4f}")
    
    print("\n✅ All tests passed!")
