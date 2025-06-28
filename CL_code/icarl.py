import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod


class iCaRL(ContinualLearningMethod):
    def __init__(self, model, device, memory_size=1000):
        super().__init__(model, device)
        self.memory_size = memory_size
        self.memory = []
        self.exemplars = []

    def _select_exemplars(self, X, y, n_exemplars):
        # 使用K-means选择最具代表性的样本
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_exemplars)
        kmeans.fit(X)

        exemplars = []
        for i in range(n_exemplars):
            cluster_samples = X[kmeans.labels_ == i]
            if len(cluster_samples) > 0:
                # 选择距离聚类中心最近的样本
                distances = np.linalg.norm(cluster_samples - kmeans.cluster_centers_[i], axis=1)
                exemplar_idx = np.argmin(distances)
                exemplars.append(cluster_samples[exemplar_idx])

        return np.array(exemplars)

    def train(self, task_id, X_task, y_task):
        # 选择样本
        n_exemplars = min(self.memory_size // (task_id + 1), len(X_task))
        exemplars = self._select_exemplars(X_task, y_task, n_exemplars)
        self.exemplars.append(exemplars)

        # 准备训练数据
        X_train = X_task
        y_train = y_task

        if self.exemplars:
            X_memory = np.vstack(self.exemplars)
            y_memory = np.zeros(len(X_memory))  # 使用0作为记忆标签
            X_train = np.vstack([X_train, X_memory])
            y_train = np.concatenate([y_train, y_memory])

        dataloader = self.prepare_dataloader(X_train, y_train)
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