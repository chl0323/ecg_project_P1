import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from torch.utils.data import TensorDataset, DataLoader

# 1. 加载深度特征
print("加载深度特征嵌入...")
try:
    embeddings = np.load('processed_data/ecg_embeddings.npy')
    labels = np.load('processed_data/ecg_labels.npy')
    print(f"成功加载特征嵌入: {embeddings.shape}")
    print(f"成功加载标签: {labels.shape}")
    print(f"标签分布: {np.bincount(labels.astype(int))}")
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    print("请先运行 new_train_transformer2.py 生成特征嵌入文件")
    exit(1)
except Exception as e:
    print(f"加载文件时出错: {e}")
    exit(1)

# 2. 随机划分为N个任务
N_TASKS = 5
indices = np.arange(embeddings.shape[0])
np.random.shuffle(indices)
split_indices = np.array_split(indices, N_TASKS)

task_datasets = []
for idx in split_indices:
    X_task = embeddings[idx]
    y_task = labels[idx]
    task_datasets.append((X_task, y_task))


# 3. 定义MLP模型
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


# 4. EWC实现
class EWC:
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if
                  p.requires_grad}
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

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                loss += _loss.sum()
        return loss


# 5. 训练和评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = embeddings.shape[1]
model = MLP(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
ewc_lambda = 5000  # EWC正则强度

ewc_list = []
acc_matrix = np.zeros((N_TASKS, N_TASKS))
f1_matrix = np.zeros((N_TASKS, N_TASKS))
auc_matrix = np.zeros((N_TASKS, N_TASKS))
recall_matrix = np.zeros((N_TASKS, N_TASKS))

for task_id, (X_task, y_task) in enumerate(task_datasets):
    print(f"\n训练任务 {task_id + 1}/{N_TASKS}")
    X_tensor = torch.tensor(X_task, dtype=torch.float32)
    y_tensor = torch.tensor(y_task, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 训练
    for epoch in range(50):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = nn.BCEWithLogitsLoss()(output, y)
            if ewc_list:
                ewc_loss = sum([ewc.penalty(model) for ewc in ewc_list])
                loss += ewc_lambda * ewc_loss
            loss.backward()
            optimizer.step()

    ewc_list.append(EWC(model, loader, device=device))

    # 评估
    for eval_id, (X_eval, y_eval) in enumerate(task_datasets[:task_id + 1]):
        model.eval()
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_eval_tensor).cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs > 0.5).astype(int)

            # 计算各项指标
            acc = accuracy_score(y_eval, preds)
            f1 = f1_score(y_eval, preds)
            auc = roc_auc_score(y_eval, probs)
            recall = recall_score(y_eval, preds)

            # 存储指标
            acc_matrix[task_id, eval_id] = acc
            f1_matrix[task_id, eval_id] = f1
            auc_matrix[task_id, eval_id] = auc
            recall_matrix[task_id, eval_id] = recall

            print(f"Task {task_id + 1} 训练后在 Task {eval_id + 1} 上 accuracy: {acc:.4f}")
            print(f"Task {task_id + 1} 训练后在 Task {eval_id + 1} 上 f1: {f1:.4f}")
            print(f"Task {task_id + 1} 训练后在 Task {eval_id + 1} 上 auc: {auc:.4f}")
            print(f"Task {task_id + 1} 训练后在 Task {eval_id + 1} 上 recall: {recall:.4f}")

# 计算每个指标的平均性能
print("\n平均性能:")
print(f"Accuracy: {np.mean(np.diag(acc_matrix)):.4f}")
print(f"F1: {np.mean(np.diag(f1_matrix)):.4f}")
print(f"AUC: {np.mean(np.diag(auc_matrix)):.4f}")
print(f"Recall: {np.mean(np.diag(recall_matrix)):.4f}")

# 保存结果
print("\n保存结果...")
os.makedirs('results', exist_ok=True)
np.save('results/acc_matrix.npy', acc_matrix)
np.save('results/f1_matrix.npy', f1_matrix)
np.save('results/auc_matrix.npy', auc_matrix)
np.save('results/recall_matrix.npy', recall_matrix)
print("结果已保存到 results 目录")