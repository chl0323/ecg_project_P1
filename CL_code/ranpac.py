import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod


class RanPAC(ContinualLearningMethod):
    def __init__(self, model, device, projection_dim=128, lambda_=1000):
        super().__init__(model, device)
        self.projection_dim = projection_dim
        self.lambda_ = lambda_
        self.projections = []
        self.params_list = []

    def _create_random_projection(self, feature_dim):
        return nn.Linear(feature_dim, self.projection_dim).to(self.device)

    def _compute_projection_loss(self, x):
        loss = 0
        # 获取当前特征
        current_features = self.model.net[0](x)  # 获取第一层特征
        feature_dim = current_features.shape[1]  # 获取特征维度

        for proj, params in zip(self.projections, self.params_list):
            # 确保投影矩阵的输入维度与特征维度匹配
            if proj.in_features != feature_dim:
                proj = nn.Linear(feature_dim, self.projection_dim).to(self.device)
                self.projections[-1] = proj

            # 计算投影
            projected_features = proj(current_features)
            # 从保存的特征中随机选择与当前批次大小相同的样本
            indices = torch.randperm(params['features'].size(0))[:x.size(0)]
            target_features = proj(params['features'][indices])
            # 计算MSE损失
            loss += torch.mean((projected_features - target_features) ** 2)
        return loss

    def train(self, task_id, X_task, y_task):
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # 获取特征维度
        with torch.no_grad():
            X_tensor = torch.tensor(X_task, dtype=torch.float32).to(self.device)
            features = self.model.net[0](X_tensor)
            feature_dim = features.shape[1]

        # 创建新的随机投影
        projection = self._create_random_projection(feature_dim)
        self.projections.append(projection)

        # 保存当前任务的特征
        self.params_list.append({'features': features.detach()})

        for epoch in range(50):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # 前向传播
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)

                # 计算投影损失
                if self.projections:
                    proj_loss = self._compute_projection_loss(x)
                    loss += self.lambda_ * proj_loss

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