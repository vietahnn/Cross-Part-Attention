# utils


import logging
import torch
import torch.nn.functional as F
import time
from statistics import mean


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, use_ensemble=False, ensemble_criterion=None):
    """
    Train for one epoch.
    
    Args:
        model: Model to train (Siformer or EnsembleModel)
        dataloader: Training data loader
        criterion: Loss criterion (used only if not ensemble)
        optimizer: Optimizer
        device: Device to use
        scheduler: Learning rate scheduler (optional)
        use_ensemble: Whether using ensemble model
        ensemble_criterion: EnsembleLoss instance (required if use_ensemble=True)
    """
    pred_correct, pred_all = 0, 0
    running_loss = 0.0
    train_time_sec_list = []
    
    # For ensemble loss tracking
    ensemble_losses = {'total': [], 'ensemble': [], 'siformer': [], 'densenet': []}
    
    for i, data in enumerate(dataloader):
        l_hands, r_hands, bodies, labels = data
        l_hands = l_hands.to(device)
        r_hands = r_hands.to(device)
        bodies = bodies.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        start_time = time.time()

        if use_ensemble:
            # Ensemble model returns (ensemble_output, siformer_output, densenet_output)
            ensemble_output, siformer_output, densenet_output = model(l_hands, r_hands, bodies, training=True)
            outputs = ensemble_output  # Use ensemble output for accuracy
            
            # Use ensemble loss
            loss, loss_dict = ensemble_criterion(ensemble_output, siformer_output, densenet_output, labels.squeeze(1))
            
            # Track losses
            for key in ensemble_losses:
                ensemble_losses[key].append(loss_dict[key])
        else:
            # Standard Siformer model
            outputs = model(l_hands, r_hands, bodies, training=True)
            loss = criterion(outputs, labels.squeeze(1))

        end_time = time.time()
        train_time_sec = end_time - start_time
        train_time_sec_list.append(train_time_sec)

        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        pred_correct += torch.sum(preds == labels.view(-1)).item()
        pred_all += labels.size(0)

    if scheduler:
        scheduler.step()

    avg_train_time = mean(train_time_sec_list)
    
    # Print ensemble loss details if applicable
    if use_ensemble and len(ensemble_losses['total']) > 0:
        avg_losses = {k: mean(v) for k, v in ensemble_losses.items()}
        print(f"  Loss breakdown: Total={avg_losses['total']:.4f}, "
              f"Ensemble={avg_losses['ensemble']:.4f}, "
              f"Siformer={avg_losses['siformer']:.4f}, "
              f"DenseNet={avg_losses['densenet']:.4f}")

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all), avg_train_time


def evaluate(model, dataloader, device, print_stats=False, use_ensemble=False):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        print_stats: Print per-class statistics
        use_ensemble: Whether using ensemble model
    """
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(100)}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)
            r_hands = r_hands.to(device)
            bodies = bodies.to(device)
            labels = labels.to(device, dtype=torch.long)

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)
                r_hand = r_hands[j].unsqueeze(0)
                body = bodies[j].unsqueeze(0)
                label = labels[j]

                if use_ensemble:
                    # Ensemble model returns (ensemble_output, siformer_output, densenet_output)
                    ensemble_output, _, _ = model(l_hand, r_hand, body, training=False)
                    output = ensemble_output.unsqueeze(0).expand(1, -1, -1)
                else:
                    output = model(l_hand, r_hand, body, training=False)
                    output = output.unsqueeze(0).expand(1, -1, -1)

                # Statistics
                if int(torch.argmax(torch.nn.functional.softmax(output, dim=2))) == int(label):
                    stats[int(labels[0][0])][0] += 1
                    pred_correct += 1

                stats[int(labels[0][0])][1] += 1
                pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return pred_correct, pred_all, (pred_correct / pred_all)


def evaluate_top_k(model, dataloader, device, k=5, use_ensemble=False):
    """
    Evaluate top-k accuracy.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        k: Top-k value
        use_ensemble: Whether using ensemble model
    """
    pred_correct, pred_all = 0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)
            r_hands = r_hands.to(device)
            bodies = bodies.to(device)
            labels = labels.to(device, dtype=torch.long)

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)
                r_hand = r_hands[j].unsqueeze(0)
                body = bodies[j].unsqueeze(0)
                label = labels[j]

                if use_ensemble:
                    ensemble_output, _, _ = model(l_hand, r_hand, body, training=False)
                    output = ensemble_output.unsqueeze(0).expand(1, -1, -1)
                else:
                    output = model(l_hand, r_hand, body, training=False)
                    output = output.unsqueeze(0).expand(1, -1, -1)

                # Statistics
                if int(label[0][0]) in torch.topk(output, k).indices.tolist():
                    pred_correct += 1

                pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)


def get_sequence_list(num):
    if num == 0:
        return [0]

    result, i = [1], 2
    while sum(result) != num:
        if sum(result) + i > num:
            for j in range(i - 1, 0, -1):
                if sum(result) + j <= num:
                    result.append(j)
        else:
            result.append(i)
        i += 1

    return sorted(result, reverse=True)