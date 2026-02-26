# utils


import logging
import torch
import torch.nn.functional as F
import time
from statistics import mean


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    embeddings = F.normalize(embeddings, dim=1)
    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=embeddings.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    return loss


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    use_contrastive=True,
    contrastive_weight=0.5,
    contrastive_temp=0.1,
    contrastive_noise_std=0.002
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

        if use_contrastive:
            noise_l1 = torch.randn_like(l_hands) * contrastive_noise_std
            noise_r1 = torch.randn_like(r_hands) * contrastive_noise_std
            noise_b1 = torch.randn_like(bodies) * contrastive_noise_std
            noise_l2 = torch.randn_like(l_hands) * contrastive_noise_std
            noise_r2 = torch.randn_like(r_hands) * contrastive_noise_std
            noise_b2 = torch.randn_like(bodies) * contrastive_noise_std

            l1, r1, b1 = l_hands + noise_l1, r_hands + noise_r1, bodies + noise_b1
            l2, r2, b2 = l_hands + noise_l2, r_hands + noise_r2, bodies + noise_b2

            outputs, emb1 = model(l1, r1, b1, training=True, return_embedding=True)
            _, emb2 = model(l2, r2, b2, training=True, return_embedding=True)
        else:
            outputs = model(l_hands, r_hands, bodies, training=True)

        end_time = time.time()
        train_time_sec = end_time - start_time
        train_time_sec_list.append(train_time_sec)

        ce_loss = criterion(outputs, labels.squeeze(1))
        if use_contrastive:
            labels_contrast = labels.squeeze(1).repeat(2)
            embeddings = torch.cat([emb1, emb2], dim=0)
            contrastive_loss = supervised_contrastive_loss(
                embeddings,
                labels_contrast,
                temperature=contrastive_temp
            )
            loss = ce_loss + (contrastive_weight * contrastive_loss)
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