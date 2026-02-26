"""
Training utilities for contrastive learning.
"""

import logging
import torch
import torch.nn.functional as F
import time
from statistics import mean


def train_epoch_with_contrastive(model, dataloader, criterion, optimizer, device, 
                                 contrastive_loss=None, scheduler=None):
    """
    Training epoch with optional contrastive/center loss.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Main classification loss (CrossEntropyLoss)
        optimizer: Optimizer
        device: Device to use
        contrastive_loss: Optional HybridContrastiveCenterLoss module
        scheduler: Optional learning rate scheduler
    
    Returns:
        Tuple of (running_loss, pred_correct, pred_all, accuracy, avg_time, loss_dict)
    """
    pred_correct, pred_all = 0, 0
    running_loss = 0.0
    running_ce_loss = 0.0
    running_con_loss = 0.0
    running_center_loss = 0.0
    train_time_sec_list = []
    
    for i, data in enumerate(dataloader):
        l_hands, r_hands, bodies, labels = data
        l_hands = l_hands.to(device)
        r_hands = r_hands.to(device)
        bodies = bodies.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        start_time = time.time()

        # Forward pass with feature extraction
        if contrastive_loss is not None:
            outputs, features = model(l_hands, r_hands, bodies, training=True, return_features=True)
        else:
            outputs = model(l_hands, r_hands, bodies, training=True)

        end_time = time.time()
        train_time_sec = end_time - start_time
        train_time_sec_list.append(train_time_sec)

        # Compute classification loss
        ce_loss = criterion(outputs, labels.squeeze(1))
        
        # Compute contrastive/center loss if enabled
        total_loss = ce_loss
        con_loss = torch.tensor(0.0).to(device)
        center_loss = torch.tensor(0.0).to(device)
        
        if contrastive_loss is not None:
            aux_losses = contrastive_loss(features, labels.squeeze(1))
            total_loss = ce_loss + aux_losses['total']
            con_loss = aux_losses['contrastive']
            center_loss = aux_losses['center']
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        running_ce_loss += ce_loss.item()
        running_con_loss += con_loss.item() if isinstance(con_loss, torch.Tensor) else con_loss
        running_center_loss += center_loss.item() if isinstance(center_loss, torch.Tensor) else center_loss

        # Statistics
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        pred_correct += torch.sum(preds == labels.view(-1)).item()
        pred_all += labels.size(0)

    if scheduler:
        scheduler.step()

    avg_train_time = mean(train_time_sec_list)
    accuracy = pred_correct / pred_all
    
    loss_dict = {
        'total': running_loss / len(dataloader),
        'ce': running_ce_loss / len(dataloader),
        'contrastive': running_con_loss / len(dataloader),
        'center': running_center_loss / len(dataloader)
    }

    return running_loss, pred_correct, pred_all, accuracy, avg_train_time, loss_dict
