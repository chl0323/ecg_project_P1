import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod


class Replay(ContinualLearningMethod):
    def __init__(self, model, device, memory_size=1000):
        super().__init__(model, device)
        self.memory_size = memory_size
        self.memory = []

    def update_memory(self, X_task, y_task):
        # 随机选择样本存入记忆
        indices = np.random.choice(len(X_task),
                                   min(self.memory_size, len(X_task)),
                                   replace=False)
        # 确保记忆中的样本保持正确的维度
        self.memory = [(X_task[i].reshape(1, -1), y_task[i]) for i in indices]

    def train(self, task_id, X_task, y_task):
        # 更新记忆
        self.update_memory(X_task, y_task)

        # 准备数据
        if self.memory:
            X_memory = np.vstack([x for x, _ in self.memory])
            y_memory = np.array([y for _, y in self.memory])
            X_combined = np.vstack([X_task, X_memory])
            y_combined = np.concatenate([y_task, y_memory])
        else:
            X_combined = X_task
            y_combined = y_task

        dataloader = self.prepare_dataloader(X_combined, y_combined)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(50):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)
                loss.backward()
                optimizer.step()

    def evaluate(self, X_eval, y_eval):
        self.model.eval()
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_eval_tensor).cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_eval, preds),
            'f1': f1_score(y_eval, preds),
            'auc': roc_auc_score(y_eval, probs),
            'recall': recall_score(y_eval, preds)
        }