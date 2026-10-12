import math
import time

import torch

from pytorchsetup import TRAIN_SIZE, batch_size, train_data, train_labels


def _sync():
    # CUDA ops run asynchronously; wait for them so the timings are real.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_timed(model, criterion, optimizer, epoch, timing_stats,
epoch_losses):
    model.train()
    epoch_loss = 0.0
    iters_per_epoch = math.ceil(TRAIN_SIZE / batch_size)

    for i in range(iters_per_epoch):
        data_start = time.time()
        data = train_data[i * batch_size : (i + 1) * batch_size]
        target = train_labels[i * batch_size : (i + 1) * batch_size]
        data_end = time.time()
        timing_stats['data_loading'] += data_end - data_start

        optimizer.zero_grad()

        forward_start = time.time()
        outputs = model(data)
        _sync()
        forward_end = time.time()
        timing_stats['forward'] += forward_end - forward_start

        loss_start = time.time()
        loss = criterion(outputs, target)
        epoch_loss += loss.item()
        loss_end = time.time()
        timing_stats['loss_computation'] += loss_end - loss_start

        backward_start = time.time()
        loss.backward()
        _sync()
        backward_end = time.time()
        timing_stats['backward'] += backward_end - backward_start

        update_start = time.time()
        optimizer.step()
        _sync()
        update_end = time.time()
        timing_stats['weight_updates'] += update_end - update_start

    epoch_losses.append(epoch_loss / iters_per_epoch)