import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod


class EWC(ContinualLearningMethod):
    def __init__(self, model, device, lambda_=5000):
        super().__init__(model, device)
        self.lambda_ = lambda_
        self.fisher_list = []  # 存储Fisher信息
        self.params_list = []  # 存储参数

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            output = self.model(x)
            loss = nn.BCEWithLogitsLoss()(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def compute_ewc_loss(self):
        loss = 0
        for fisher, params in zip(self.fisher_list, self.params_list):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    _loss = fisher[n] * (p - params[n]).pow(2)
                    loss += _loss.sum()
        return loss

    def train(self, task_id, X_task, y_task):
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(50):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)

                if self.fisher_list:  # 如果有之前的任务
                    ewc_loss = self.compute_ewc_loss()
                    loss += self.lambda_ * ewc_loss

                loss.backward()
                optimizer.step()

        # 保存当前参数和Fisher信息
        current_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        current_fisher = self._compute_fisher(dataloader)

        self.params_list.append(current_params)
        self.fisher_list.append(current_fisher)

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