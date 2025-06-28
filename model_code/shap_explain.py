import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 特征名
feature_cols = [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'heart_rate',
    'QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50'
]

# 加载模型
print("加载模型...")
model = tf.keras.models.load_model('transformer_full_model.keras', compile=False)

# 加载测试集数据
print("加载测试集数据...")
X = np.load('processed_data/X_test_smote.npy')
sequence_length = 10
n_features = X.shape[1] // sequence_length
X = X.reshape(-1, sequence_length, n_features)

# 选择背景样本和要解释的样本
background = X[np.random.choice(X.shape[0], 100, replace=False)]
explain_samples = X[:10]

# 创建SHAP解释器
print("创建SHAP解释器...")
explainer = shap.DeepExplainer(model, background)

# 计算SHAP值
print("计算SHAP值...")
shap_values = explainer.shap_values(explain_samples)

# 对时间步做平均，得到每个特征的平均shap值
shap_mean = np.mean(shap_values[0], axis=1)  # shape: (样本数, 特征数)

# 可视化
print("绘制SHAP summary plot...")
shap.summary_plot(shap_mean, feature_names=feature_cols)
plt.show() 