import pandas as pd
import numpy as np
from new_transformer_feature_extractor import TransformerFeatureExtractor
import tensorflow as tf
from keras import layers
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# 创建保存目录
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# 直接加载预处理好的数据
print("加载预处理数据...")
X_train = np.load('processed_data/X_train_smote.npy')
y_train = np.load('processed_data/y_train_smote.npy')
X_val = np.load('processed_data/X_val_smote.npy')
y_val = np.load('processed_data/y_val_smote.npy')

# 重塑数据为序列形式 (samples, sequence_length, features)
sequence_length = 10
n_features = X_train.shape[1] // sequence_length
X_train = X_train.reshape(-1, sequence_length, n_features)
X_val = X_val.reshape(-1, sequence_length, n_features)

print(f"训练数据形状: {X_train.shape}")
print(f"验证数据形状: {X_val.shape}")

feature_cols = [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
    'heart_rate', 'QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50'
]

# 构建模型
print("构建模型...")
extractor = TransformerFeatureExtractor(
    input_dim=n_features,
    sequence_length=sequence_length,
    num_heads=1,
    d_model=128,  # 使用d_model替代dff
    rate=0.1
)
model = extractor.build()
model_out = layers.Dense(1, activation='sigmoid')(model.output)
full_model = tf.keras.Model(model.input, model_out)

full_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# 设置检查点路径
checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.weights.h5')
print(f"检查点保存路径: {checkpoint_dir}")

# 自定义回调，定期保存模型
class PeriodicSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, period=1):
        super(PeriodicSaveCallback, self).__init__()
        self.save_path = save_path
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.model.save_weights(self.save_path.format(epoch=epoch + 1))
            print(f"Model saved at epoch {epoch + 1}")

# 训练
print("开始训练...")
try:
    history = full_model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            PeriodicSaveCallback(checkpoint_path, period=1)
        ]
    )
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.csv')
    pd.DataFrame(history.history).to_csv(history_path, index=False)
except Exception as e:
    print(f"Training interrupted: {e}")
    interrupted_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interrupted_model.weights.h5')
    full_model.save_weights(interrupted_model_path)
    print("Model state saved. You can resume training later.")

# 保存特征提取部分权重
print("保存模型权重...")
feature_extractor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_transformer_feature_extractor_weights.weights.h5')
model.save_weights(feature_extractor_path)

# 保存整个模型
full_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformer_full_model.keras')
full_model.save(full_model_path)
print("训练完成！")