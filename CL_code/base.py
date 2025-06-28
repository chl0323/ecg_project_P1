import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ContinualLearningMethod:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def prepare_dataloader(self, X, y, batch_size=64):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, task_id, X_task, y_task):
        raise NotImplementedError

    def evaluate(self, X_eval, y_eval):
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)