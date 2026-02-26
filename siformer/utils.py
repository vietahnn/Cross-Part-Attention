# utils


import logging
import torch
import torch.nn.functional as F
import time
from statistics import mean


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    pred_correct, pred_all = 0, 0
    running_loss = 0.0
    train_time_sec_list = []
    for i, data in enumerate(dataloader):
        l_hands, r_hands, bodies, labels = data
        l_hands = l_hands.to(device)
        r_hands = r_hands.to(device)
        bodies = bodies.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        start_time = time.time()

        outputs = model(l_hands, r_hands, bodies, training=True)

        end_time = time.time()
        train_time_sec = end_time - start_time
        train_time_sec_list.append(train_time_sec)

        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        # print(f'preds: {preds}')
        # print(f'label: {labels.view(-1)}')
        pred_correct += torch.sum(preds == labels.view(-1)).item()
        pred_all += labels.size(0)

    if scheduler:
        scheduler.step()

    avg_train_time = mean(train_time_sec_list)

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all), avg_train_time


def evaluate(model, dataloader, device, print_stats=False):
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(100)}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)  # [24, 204, 21, 2]
            r_hands = r_hands.to(device)  # [24, 204, 21, 2]
            bodies = bodies.to(device)  # [24, 204, 12, 2]
            labels = labels.to(device, dtype=torch.long)  # [24, 1]

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

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


def evaluate_top_k(model, dataloader, device, k=5):
    pred_correct, pred_all = 0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)
            r_hands = r_hands.to(device)
            bodies = bodies.to(device)
            labels = labels.to(device, dtype=torch.long)

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

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

def train_epoch_with_contrastive(model, dataloader, criterion, optimizer, device, contrastive_loss=None, scheduler=None):
    pred_correct, pred_all = 0, 0
    running_loss, running_ce_loss, running_con_loss, running_center_loss = 0.0, 0.0, 0.0, 0.0
    train_time_sec_list = []
    for i, data in enumerate(dataloader):
        l_hands, r_hands, bodies, labels = data
        l_hands, r_hands, bodies = l_hands.to(device), r_hands.to(device), bodies.to(device)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        start_time = time.time()
        if contrastive_loss is not None:
            outputs, features = model(l_hands, r_hands, bodies, training=True, return_features=True)
        else:
            outputs = model(l_hands, r_hands, bodies, training=True)
        end_time = time.time()
        train_time_sec_list.append(end_time - start_time)
        ce_loss = criterion(outputs, labels.squeeze(1))
        total_loss = ce_loss
        con_loss = torch.tensor(0.0).to(device)
        center_loss = torch.tensor(0.0).to(device)
        if contrastive_loss is not None:
            aux_losses = contrastive_loss(features, labels.squeeze(1))
            total_loss = ce_loss + aux_losses['total']
            con_loss, center_loss = aux_losses['contrastive'], aux_losses['center']
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        running_ce_loss += ce_loss.item()
        running_con_loss += con_loss.item() if isinstance(con_loss, torch.Tensor) else con_loss
        running_center_loss += center_loss.item() if isinstance(center_loss, torch.Tensor) else center_loss
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        pred_correct +torch.sum(preds == labels.view(-1)).item()
        pred_all += labels.size(0)
    if scheduler:
        scheduler.step()
    avg_train_time = mean(train_time_sec_list)
    accuracy = pred_correct / pred_all
    loss_dict = {'total': running_loss / len(dataloader), 'ce': running_ce_loss / len(dataloader), 'contrastive': running_con_loss / len(dataloader), 'center': running_center_loss / len(dataloader)}
    return running_loss, pred_correct, pred_all, accuracy, avg_train_time, loss_dict
