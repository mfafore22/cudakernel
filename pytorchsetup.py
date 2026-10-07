import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

TRAIN_SIZE = 10000
epochs = 10
learning_rate = 1e-2
batch_size = 8

torch.set_float32_matmul_precision("high")

X_train_np = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(
60000, 784)
y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)
X_test_np = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(
10000, 784)
y_test_np = np.fromfile("data/y_test.bin", dtype=np.int32)

mean, std = 0.1307, 0.3081
X_train_np = (X_train_np - mean) / std
X_test_np = (X_test_np - mean) / std

train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape(
-1, 1, 28, 28)).to("cuda")
train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda")
test_data = torch.from_numpy(X_test_np.reshape(-1, 1, 28, 28)).to("cuda")
test_labels = torch.from_numpy(y_test_np).long().to("cuda")