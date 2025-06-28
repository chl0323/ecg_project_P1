import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from new_transformer_feature_extractor import TransformerFeatureExtractor
import os

class TransformerModel(nn.Module):
    def __init__(self, d_model=512, num_heads=4, num_layers=6, dim_feedforward=2048, dropout=0.1, sequence_length=10):
        super().__init__()
        self.feature_extractor = TransformerFeatureExtractor(
            input_dim=17,  # 假设输入特征维度为17
            sequence_length=sequence_length,
            num_heads=num_heads,
            d_model=d_model,
            rate=dropout
        )
        self.decoder = nn.Linear(d_model, 1)  # 假设输出维度为1
        self.best_metric = float('inf')  # 用于跟踪最佳指标
        self.best_model_path = 'best_model.pt'  # 保存最佳模型的路径

    def forward(self, src):
        features = self.feature_extractor.build()(src)
        output = self.decoder(features)
        return output

    def save_best_model(self, metric):
        if metric < self.best_metric:
            self.best_metric = metric
            torch.save(self.state_dict(), self.best_model_path)
            print(f"保存最佳模型，指标: {metric}")

    def train_model(self, train_loader, val_loader, num_epochs, optimizer, criterion):
        for epoch in range(num_epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self(batch[0])
                loss = criterion(outputs, batch[1])
                loss.backward()
                optimizer.step()
            
            # 验证
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self(batch[0])
                    val_loss += criterion(outputs, batch[1]).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            self.save_best_model(val_loss)

if __name__ == "__main__":
    import numpy as np
    X_train = np.load('X_train.npy')  # shape: (N, seq_len, 17)
    y_train = np.load('y_train.npy')  # shape: (N, 1)
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    model = TransformerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()  # 二分类

    model.train_model(train_loader, val_loader, num_epochs=50, optimizer=optimizer, criterion=criterion) 