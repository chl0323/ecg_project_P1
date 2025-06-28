import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod


class LwF(ContinualLearningMethod):
    def __init__(self, model, device, temperature=2.0, lambda_=1.0):
        super().__init__(model, device)
        self.temperature = temperature
        self.lambda_ = lambda_
        self.old_model = None

    def _distill_loss(self, new_logits, old_logits):
        # 确保logits维度正确
        new_logits = new_logits.unsqueeze(-1)  # [batch_size, 1]
        old_logits = old_logits.unsqueeze(-1)  # [batch_size, 1]

        # 计算KL散度损失
        return nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(new_logits / self.temperature, dim=0),
            torch.softmax(old_logits / self.temperature, dim=0)
        ) * (self.temperature ** 2)

    def train(self, task_id, X_task, y_task):
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # 保存旧模型
        if self.old_model is None:
            self.old_model = type(self.model)(self.model.net[0].in_features).to(self.device)
        self.old_model.load_state_dict(self.model.state_dict())

        for epoch in range(50):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # 新任务损失
                new_logits = self.model(x)
                loss = nn.BCEWithLogitsLoss()(new_logits, y)

                # 知识蒸馏损失
                if task_id > 0:
                    with torch.no_grad():
                        old_logits = self.old_model(x)
                    distill_loss = self._distill_loss(new_logits, old_logits)
                    loss += self.lambda_ * distill_loss

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