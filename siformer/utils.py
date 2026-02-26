# utils


import logging
import torch
import torch.nn.functional as F
import time
from statistics import mean


def temporal_crop_and_resize(x, min_ratio=0.7, max_ratio=1.0):
    # x: [B, T, J, C]
    batch_size, seq_len, joints, coords = x.shape
    crop_ratio = torch.empty(1, device=x.device).uniform_(min_ratio, max_ratio).item()
    crop_len = max(1, int(seq_len * crop_ratio))
    start = 0 if crop_len == seq_len else torch.randint(0, seq_len - crop_len + 1, (1,)).item()
    cropped = x[:, start:start + crop_len, :, :]

    if crop_len == seq_len:
        return cropped

    reshaped = cropped.permute(0, 2, 3, 1).contiguous().view(batch_size, joints * coords, crop_len)
    resized = F.interpolate(reshaped, size=seq_len, mode="linear", align_corners=False)
    resized = resized.view(batch_size, joints, coords, seq_len).permute(0, 3, 1, 2).contiguous()
    return resized


def temporal_invariance_loss(logits_a, logits_b, temperature=2.0):
    probs_a = F.softmax(logits_a / temperature, dim=1)
    probs_b = F.softmax(logits_b / temperature, dim=1)
    log_probs_a = F.log_softmax(logits_a / temperature, dim=1)
    log_probs_b = F.log_softmax(logits_b / temperature, dim=1)

    kl_ab = F.kl_div(log_probs_a, probs_b.detach(), reduction="batchmean")
    kl_ba = F.kl_div(log_probs_b, probs_a.detach(), reduction="batchmean")
    return (kl_ab + kl_ba) * (temperature ** 2)


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    use_temporal_invariance=True,
    temporal_weight=0.5,
    temporal_temp=2.0,
    temporal_min_ratio=0.7,
    temporal_max_ratio=1.0
):
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
        temporal_loss = None
        if use_temporal_invariance:
            l_view1 = temporal_crop_and_resize(l_hands, temporal_min_ratio, temporal_max_ratio)
            r_view1 = temporal_crop_and_resize(r_hands, temporal_min_ratio, temporal_max_ratio)
            b_view1 = temporal_crop_and_resize(bodies, temporal_min_ratio, temporal_max_ratio)
            l_view2 = temporal_crop_and_resize(l_hands, temporal_min_ratio, temporal_max_ratio)
            r_view2 = temporal_crop_and_resize(r_hands, temporal_min_ratio, temporal_max_ratio)
            b_view2 = temporal_crop_and_resize(bodies, temporal_min_ratio, temporal_max_ratio)

            logits_view1 = model(l_view1, r_view1, b_view1, training=True)
            logits_view2 = model(l_view2, r_view2, b_view2, training=True)
            temporal_loss = temporal_invariance_loss(logits_view1, logits_view2, temperature=temporal_temp)

        end_time = time.time()
        train_time_sec = end_time - start_time
        train_time_sec_list.append(train_time_sec)

        ce_loss = criterion(outputs, labels.squeeze(1))
        if temporal_loss is not None:
            loss = ce_loss + (temporal_weight * temporal_loss)
        else:
            loss = ce_loss
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