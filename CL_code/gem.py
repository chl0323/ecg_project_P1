import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod


class GEM(ContinualLearningMethod):
    def __init__(self, model, device, memory_size=1000):
        super().__init__(model, device)
        self.memory_size = memory_size
        self.memory = []
        self.grad_dims = []
        self.grads = []

    def _compute_grad(self, x, y):
        self.model.zero_grad()
        output = self.model(x)
        loss = nn.BCEWithLogitsLoss()(output, y)
        loss.backward()
        grad = []
        for p in self.model.parameters():
            if p.grad is not None:
                grad.append(p.grad.data.view(-1))
        return torch.cat(grad)

    def _project_grad(self, grad):
        if not self.grads:
            return grad

        for g in self.grads:
            dot_product = torch.dot(grad, g)
            if dot_product < 0:
                grad -= (dot_product / torch.dot(g, g)) * g
        return grad

    def train(self, task_id, X_task, y_task):
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # 更新记忆
        indices = np.random.choice(len(X_task), min(self.memory_size, len(X_task)), replace=False)
        self.memory = [(X_task[i], y_task[i]) for i in indices]

        for epoch in range(50):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                # 计算当前梯度
                grad = self._compute_grad(x, y)

                # 投影梯度
                if self.grads:
                    grad = self._project_grad(grad)

                # 更新参数
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)
                loss.backward()

                # 应用投影后的梯度
                idx = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data = grad[idx:idx + p.grad.numel()].view(p.grad.shape)
                        idx += p.grad.numel()

                optimizer.step()

            # 保存梯度
            if task_id > 0:
                self.grads.append(grad.detach())

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