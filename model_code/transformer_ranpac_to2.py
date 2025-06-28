import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from drift_detection_module import DDM

class TransformerRanPAC:
    def __init__(self, n_trials=50, input_dim=None, ridge=1e3, M=1000, class_weight=None):
        self.n_trials = n_trials
        self.best_params = None
        self.best_transformer = None
        self.best_rf = None
        self.scaler = StandardScaler()
        self.ridge = ridge
        self.M = M
        self.input_dim = input_dim
        self.W_rand = None
        self.Q = None
        self.G = None
        self.drift_detector = DDM()
        self.simplex = {
            'best': {'ridge': 1e-3, 'error': []},
            'worst': {'ridge': 1e3, 'error': []}
        }
        self.nelder_mead_operator = 0
        self.is_search_phase = True
        self.num_data = 0
        self.total_samples_processed = 0
        self.checkpoint_interval = 10000
        self.feature_columns =[
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
    'heart_rate', 'QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50'
]
        self.class_weight = class_weight

    def save_checkpoint(self, filepath):
        checkpoint = {
            'W_rand': self.W_rand,
            'Q': self.Q,
            'G': self.G,
            'ridge': self.ridge,
            'simplex': self.simplex,
            'total_samples_processed': self.total_samples_processed,
            'scaler': self.scaler
        }
        np.save(filepath, checkpoint)
        print(f"检查点已保存到: {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = np.load(filepath, allow_pickle=True).item()
        self.W_rand = checkpoint['W_rand']
        self.Q = checkpoint['Q']
        self.G = checkpoint['G']
        self.ridge = checkpoint['ridge']
        self.simplex = checkpoint['simplex']
        self.total_samples_processed = checkpoint['total_samples_processed']
        self.scaler = checkpoint['scaler']
        print(f"已从检查点恢复: {filepath}")

    def setup_random_projection(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be set before setup_random_projection")
        self.W_rand = np.random.randn(self.input_dim, self.M) / np.sqrt(self.M)
        self.Q = np.zeros((self.M, 2))  # 2类
        self.G = np.zeros((self.M, self.M))

    def nelder_mead_optimize(self):
        if len(self.simplex['best']['error']) < 30 or len(self.simplex['worst']['error']) < 30:
            return
        best_error = np.mean(self.simplex['best']['error'])
        worst_error = np.mean(self.simplex['worst']['error'])
        if best_error == worst_error:
            return
        best_ridge = np.log10(self.simplex['best']['ridge'])
        worst_ridge = np.log10(self.simplex['worst']['ridge'])
        centroid = (best_ridge + worst_ridge) / 2
        reflection = 2 * centroid - worst_ridge
        self.simplex['worst']['ridge'] = 10 ** reflection
        self.simplex['worst']['error'] = []
        self.num_data = 0

    def update_model(self, x, y):
        if self.W_rand is None:
            self.input_dim = x.shape[1]
            self.setup_random_projection()
        x_proj = np.maximum(0, x @ self.W_rand)
        y_onehot = np.zeros((len(y), 2))
        y_onehot[np.arange(len(y)), y] = 1
        if self.class_weight is not None:
            sample_weights = np.array([self.class_weight[label] for label in y])
            y_onehot = y_onehot * sample_weights[:, None]
        self.Q += x_proj.T @ y_onehot
        self.G += x_proj.T @ x_proj
        if self.is_search_phase:
            self.num_data += 1
            for key in self.simplex:
                ridge = self.simplex[key]['ridge']
                W = np.linalg.solve(
                    self.G + ridge * np.eye(self.G.shape[0]),
                    self.Q
                ).T
                output = x_proj @ W.T
                pred = np.argmax(output, axis=1)
                error = (pred != y).astype(float)
                self.simplex[key]['error'].extend(error)
            self.nelder_mead_optimize()
            self.ridge = self.simplex['best']['ridge']
        error = 0
        self.drift_detector.update(error)
        if self.drift_detector.drift_detected:
            self.reset_model()

    def reset_model(self):
        self.Q = np.zeros((self.M, 2))
        self.G = np.zeros((self.M, self.M))
        self.is_search_phase = True
        self.num_data = 0
        for key in self.simplex:
            self.simplex[key]['error'] = []

    def fit(self, data, labels, task_id=None, checkpoint_dir='checkpoints'):
        import os
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if isinstance(data, pd.DataFrame):
            features = data[self.feature_columns].values
        else:
            features = data
        X_processed = self.scaler.fit_transform(features)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, labels, test_size=0.1, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        print("开始在线学习...")
        batch_size = 512
        n_samples = len(X_train)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        try:
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                self.update_model(X_batch, y_batch)
                self.total_samples_processed += len(batch_indices)
                if self.total_samples_processed % self.checkpoint_interval == 0:
                    progress = (i + len(batch_indices)) / n_samples * 100
                    print(f"进度: {progress:.2f}% ({self.total_samples_processed} 样本)")
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'checkpoint_{self.total_samples_processed}.npy'
                    )
                    self.save_checkpoint(checkpoint_path)
                    if len(X_val) > 0:
                        val_pred = self.predict(X_val[:1000])
                        val_acc = np.mean(val_pred == y_val[:1000])
                        print(f"验证集准确率: {val_acc:.4f}")
        except KeyboardInterrupt:
            print("\n训练被中断，保存最后的检查点...")
            final_checkpoint_path = os.path.join(checkpoint_dir, 'interrupted_checkpoint.npy')
            self.save_checkpoint(final_checkpoint_path)
            return self
        final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.npy')
        self.save_checkpoint(final_checkpoint_path)
        print("训练完成！")
        return self

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            features = data[self.feature_columns].values
        else:
            features = data
        X_processed = self.scaler.transform(features)
        X_proj = np.maximum(0, X_processed @ self.W_rand)
        W = np.linalg.solve(
            self.G + self.ridge * np.eye(self.G.shape[0]),
            self.Q
        ).T
        output = X_proj @ W.T
        return np.argmax(output, axis=1)

    def evaluate(self, data, labels):
        y_pred = self.predict(data)
        conf_matrix = confusion_matrix(labels, y_pred)
        class_report = classification_report(labels, y_pred, target_names=['No T2DM', 'T2DM'])
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['No T2DM', 'T2DM']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'plot': plt.gcf()
        }