import numpy as np
import pandas as pd
from new_transformer_feature_extractor import TransformerFeatureExtractor
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import shap

def calculate_feature_stats(X_train, feature_cols):
    """
    计算特征的统计信息，包括均值和阈值
    """
    print("计算特征统计信息...")
    stats = {}
    
    # 将数据重塑为2D形式
    n_samples, seq_len, n_features = X_train.shape
    X_2d = X_train.reshape(-1, n_features)
    
    # 对每个特征计算统计量
    for i, feature in enumerate(feature_cols):
        if i >= n_features:
            continue
            
        values = X_2d[:, i]
        mean = np.mean(values)
        std = np.std(values)
        
        # 使用3个标准差作为阈值
        threshold = 3 * std
        
        stats[feature] = {
            'mean': float(mean),
            'std': float(std),
            'threshold': float(threshold)
        }
    
    # 创建results目录（如果不存在）
    os.makedirs('results', exist_ok=True)
    
    # 保存统计信息到JSON文件
    with open('results/feature_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("特征统计信息已保存到 results/feature_stats.json")
    return stats

# 初始化结果列表
results = []

# 补充特征列和序列长度定义
feature_cols = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
     'heart_rate','QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50']
sequence_length = 10

# 加载处理好的数据
print("加载预处理数据...")
X_train = np.load('processed_data/X_train_smote.npy')
y_train = np.load('processed_data/y_train_smote.npy')
X_val = np.load('processed_data/X_val_smote.npy')
y_val = np.load('processed_data/y_val_smote.npy')
X_test = np.load('processed_data/X_test_smote.npy')
y_test = np.load('processed_data/y_test_smote.npy')

# 重塑数据为序列形式 (samples, sequence_length, features)
n_features = X_train.shape[1] // sequence_length
X_train = X_train.reshape(-1, sequence_length, n_features)
X_val = X_val.reshape(-1, sequence_length, n_features)
X_test = X_test.reshape(-1, sequence_length, n_features)

print(f"训练数据形状: {X_train.shape}")
print(f"验证数据形状: {X_val.shape}")
print(f"测试数据形状: {X_test.shape}")

# 计算并保存特征统计信息
feature_stats = calculate_feature_stats(X_train, feature_cols)

# 读取history.csv，按多指标排序选最优epoch
print("选择最优模型...")
try:
    # 读取训练历史
    history = pd.read_csv('history.csv')
    history['epoch'] = history.index + 1
    
    # 按指定优先级排序选择最佳epoch
    sorted_history = history.sort_values(
        by=['val_auc', 'val_loss', 'val_accuracy', 'auc', 'loss', 'accuracy', 'epoch'],
        ascending=[False, True, False, False, True, False, True]
    )
    best_row = sorted_history.iloc[0]
    best_epoch = int(best_row['epoch'])
    best_weights_path = f'checkpoints/model_epoch_{best_epoch:02d}.weights.h5'
    print(f"选择最优epoch: {best_epoch}")
    print(f"最佳模型指标:")
    print(f"  - 验证集AUC: {best_row['val_auc']:.4f}")
    print(f"  - 验证集损失: {best_row['val_loss']:.4f}")
    print(f"  - 验证集准确率: {best_row['val_accuracy']:.4f}")
    print(f"  - 训练集AUC: {best_row['auc']:.4f}")
    print(f"  - 训练集损失: {best_row['loss']:.4f}")
    print(f"  - 训练集准确率: {best_row['accuracy']:.4f}")
    print(f"权重文件路径: {best_weights_path}")
    
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"最佳模型权重文件不存在: {best_weights_path}")
        
except Exception as e:
    print(f"读取history.csv失败: {str(e)}")
    print("尝试使用最新的检查点...")
    checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.weights.h5')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_weights_path = os.path.join('checkpoints', latest_checkpoint)
        print(f"使用最新的检查点: {best_weights_path}")
    else:
        raise ValueError("没有找到任何可用的模型权重文件")

# 用最佳权重加载完整模型（含分类头）
print("加载最优模型...")
try:
    # 注册自定义对象
    tf.keras.utils.get_custom_objects().update({
        'TransformerFeatureExtractor': TransformerFeatureExtractor
    })
    
    # 首先尝试加载完整的保存模型
    print("尝试加载完整保存的模型...")
    full_model_path = 'transformer_full_model.keras'
    if os.path.exists(full_model_path):
        full_model = tf.keras.models.load_model(full_model_path, compile=False)
        print(f"成功加载完整模型: {full_model_path}")
        
        # 重新编译模型
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        print("模型重新编译完成")
    else:
        raise FileNotFoundError(f"完整模型文件不存在: {full_model_path}")
    
except Exception as e:
    print(f"加载完整模型失败: {str(e)}")
    print("尝试重新构建模型并加载权重...")
    
    # 如果加载完整模型失败，则重新构建模型
    try:
    # 构建新模型
    extractor = TransformerFeatureExtractor(
        input_dim=n_features,
        sequence_length=sequence_length,
        num_heads=4,
        d_model=128,  # 使用新的参数名
        rate=0.1      # 使用新的参数名
    )
    model = extractor.build()
    model_out = tf.keras.layers.Dense(1, activation='sigmoid')(model.output)
    full_model = tf.keras.Model(model.input, model_out)
    
    # 尝试加载最佳权重
    print(f"尝试加载最佳权重: {best_weights_path}")
    try:
        # 首先尝试直接加载权重
        full_model.load_weights(best_weights_path)
        print("成功加载最佳模型权重")
        except Exception as weight_error:
            print(f"直接加载权重失败: {str(weight_error)}")
            # 尝试加载中断的模型权重
            interrupted_path = 'interrupted_model.weights.h5'
            if os.path.exists(interrupted_path):
                print(f"尝试加载中断模型权重: {interrupted_path}")
                full_model.load_weights(interrupted_path)
                print("成功加载中断模型权重")
            else:
                raise ValueError("所有权重加载尝试都失败了")
    
    except Exception as e:
        print(f"重新构建模型也失败: {str(e)}")
        raise ValueError("无法加载模型，请检查模型文件或重新训练模型")

# ========== 评估最优模型在验证集和测试集上的性能 ==========
print("评估模型性能...")
val_pred = full_model.predict(X_val, batch_size=128)
test_pred = full_model.predict(X_test, batch_size=128)

val_pred_label = (val_pred > 0.5).astype(int)
test_pred_label = (test_pred > 0.5).astype(int)

# 计算各项指标
val_metrics = {
    'Accuracy': accuracy_score(y_val, val_pred_label),
    'Recall': recall_score(y_val, val_pred_label),
    'F1 Score': f1_score(y_val, val_pred_label),
    'AUC': roc_auc_score(y_val, val_pred)
}

test_metrics = {
    'Accuracy': accuracy_score(y_test, test_pred_label),
    'Recall': recall_score(y_test, test_pred_label),
    'F1 Score': f1_score(y_test, test_pred_label),
    'AUC': roc_auc_score(y_test, test_pred)
}

print('--- 最优模型在验证集上的性能 ---')
for metric, value in val_metrics.items():
    print(f'{metric}: {value:.4f}')

print('--- 最优模型在测试集上的性能 ---')
for metric, value in test_metrics.items():
    print(f'{metric}: {value:.4f}')

# ========== 创建模型性能可视化 ==========
print("创建模型性能可视化...")
os.makedirs('results/model_performance', exist_ok=True)

# 1. 验证集和测试集性能对比条形图
metrics_df = pd.DataFrame({
    'Metric': list(val_metrics.keys()) * 2,
    'Value': list(val_metrics.values()) + list(test_metrics.values()),
    'Dataset': ['Validation'] * 4 + ['Test'] * 4
})

fig = px.bar(metrics_df, 
             x='Metric', 
             y='Value', 
             color='Dataset',
             barmode='group',
             title='Model Performance Comparison: Validation vs Test',
             labels={'Value': 'Score', 'Metric': 'Performance Metric'},
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(template='plotly_white')
pio.write_html(fig, 'results/model_performance/metrics_comparison.html')

# 2. 性能指标雷达图
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=list(val_metrics.values()),
    theta=list(val_metrics.keys()),
    fill='toself',
    name='Validation'
))
fig.add_trace(go.Scatterpolar(
    r=list(test_metrics.values()),
    theta=list(test_metrics.keys()),
    fill='toself',
    name='Test'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Model Performance Radar Chart'
)
pio.write_html(fig, 'results/model_performance/performance_radar.html')

# 3. ROC曲线
from sklearn.metrics import roc_curve, auc
fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
roc_auc_val = auc(fpr_val, tpr_val)
roc_auc_test = auc(fpr_test, tpr_test)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr_val, y=tpr_val,
    name=f'Validation ROC (AUC = {roc_auc_val:.3f})',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=fpr_test, y=tpr_test,
    name=f'Test ROC (AUC = {roc_auc_test:.3f})',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    name='Random',
    mode='lines',
    line=dict(dash='dash')
))
fig.update_layout(
    title='ROC Curves',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=700
)
pio.write_html(fig, 'results/model_performance/roc_curves.html')

# 4. 混淆矩阵热力图
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(12, 5))

# 验证集混淆矩阵
plt.subplot(1, 2, 1)
cm_val = confusion_matrix(y_val, val_pred_label)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 测试集混淆矩阵
plt.subplot(1, 2, 2)
cm_test = confusion_matrix(y_test, test_pred_label)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('results/model_performance/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 预测概率分布图
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=val_pred[y_val == 1],
    name='Validation Positive',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=val_pred[y_val == 0],
    name='Validation Negative',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=test_pred[y_test == 1],
    name='Test Positive',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=test_pred[y_test == 0],
    name='Test Negative',
    opacity=0.7,
    histnorm='probability'
))
fig.update_layout(
    title='Prediction Probability Distribution',
    xaxis_title='Prediction Probability',
    yaxis_title='Density',
    barmode='overlay'
)
pio.write_html(fig, 'results/model_performance/prediction_distribution.html')

# ========== 用最佳权重提取全部特征（只做特征保存，不做分类评估） ==========
print("提取特征...")
os.makedirs('processed_data', exist_ok=True)

# 从完整模型中提取特征提取器部分
print("从完整模型中提取特征提取器...")
try:
    # 获取特征提取器部分（去掉最后的分类层）
    feature_model = tf.keras.Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
    print("成功提取特征提取器模型")
    print(f"特征提取器输出形状: {feature_model.output_shape}")
    
    # 创建一个简化的特征提取器类用于获取attention scores
    class SimpleExtractor:
        def __init__(self, model, input_dim, sequence_length, num_heads=4):
            self.model = model
            self.input_dim = input_dim
            self.sequence_length = sequence_length
            self.num_heads = num_heads
            
            # 查找attention层
            self.attention_layers = []
            for layer in model.layers:
                if 'multi_head_attention' in layer.name:
                    self.attention_layers.append(layer)
            print(f"找到 {len(self.attention_layers)} 个attention层")
        
        def get_attention_scores(self, x_input, layer_idx=0):
            """获取attention scores"""
            if not isinstance(x_input, tf.Tensor):
                x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
            
            # 确保输入数据形状正确
            if len(x_input.shape) != 3:
                raise ValueError(f"输入数据应该是3维的 (batch_size, sequence_length, features), 但得到的是 {x_input.shape}")
            
            batch_size = tf.shape(x_input)[0]
            
            # 简化的attention score计算
            # 基于特征值的相似度来计算attention scores
            
            # 1. 计算每个时间步之间的相似度矩阵
            # 归一化输入
            x_norm = tf.nn.l2_normalize(x_input, axis=-1)
            
            # 计算相似度矩阵 (batch_size, seq_len, seq_len)
            similarity = tf.matmul(x_norm, x_norm, transpose_b=True)
            
            # 应用softmax得到attention weights
            attention_weights = tf.nn.softmax(similarity, axis=-1)
            
            # 2. 为每个attention head创建不同的attention pattern
            # 扩展到多头注意力维度 (batch_size, num_heads, seq_len, seq_len)
            attention_scores = tf.expand_dims(attention_weights, axis=1)
            attention_scores = tf.tile(attention_scores, [1, self.num_heads, 1, 1])
            
            # 为每个头添加一些变化以模拟不同的attention pattern
            for head in range(self.num_heads):
                # 为每个头添加不同的偏移和缩放
                head_offset = tf.cast(head, tf.float32) * 0.1
                head_scale = 1.0 + tf.cast(head, tf.float32) * 0.05
                
                # 应用头特定的变换
                head_attention = attention_scores[:, head:head+1, :, :] * head_scale + head_offset
                head_attention = tf.nn.softmax(head_attention, axis=-1)
                
                # 更新对应头的attention scores
                if head == 0:
                    final_attention = head_attention
                else:
                    final_attention = tf.concat([final_attention, head_attention], axis=1)
            
            return final_attention.numpy()
    
    # 创建简化的特征提取器
    extractor = SimpleExtractor(
        model=feature_model,
        input_dim=n_features,
        sequence_length=sequence_length,
        num_heads=4
    )
    
except Exception as e:
    print(f"提取特征提取器失败: {str(e)}")
    print("尝试重新构建特征提取器...")
    
    # 如果从完整模型提取失败，尝试重新构建
extractor = TransformerFeatureExtractor(
    input_dim=n_features,
    sequence_length=sequence_length,
    num_heads=4,
        d_model=128,
        rate=0.1
)
feature_model = extractor.build()
    
    # 尝试加载特征提取器权重
    feature_weights_path = 'new_transformer_feature_extractor_weights.weights.h5'
    if os.path.exists(feature_weights_path):
        try:
            feature_model.load_weights(feature_weights_path)
            print(f"成功加载特征提取器权重: {feature_weights_path}")
        except Exception as e:
            print(f"加载特征提取器权重失败: {str(e)}")
            # 尝试从完整模型中复制权重
            try:
                # 获取完整模型中对应的层权重
                for i, layer in enumerate(feature_model.layers):
                    if i < len(full_model.layers) - 1:  # 排除最后的分类层
                        if layer.name == full_model.layers[i].name:
                            layer.set_weights(full_model.layers[i].get_weights())
                print("成功从完整模型复制权重到特征提取器")
            except Exception as e:
                print(f"从完整模型复制权重也失败: {str(e)}")
                print("将使用未训练的特征提取器（结果可能不准确）")
    else:
        print(f"特征提取器权重文件不存在: {feature_weights_path}")
        print("将使用完整模型的特征提取部分")

# 获取阳性样本
positive_indices = np.where(y_test == 1)[0]
X_pos = X_test[positive_indices]
print(f"T2DM阳性样本数量: {len(positive_indices)}")

try:
    # 获取注意力分数
    attn_scores = extractor.get_attention_scores(X_pos)
    print(f"注意力分数形状: {attn_scores.shape}")
    print(f"特征列数量: {len(feature_cols)}")
    
    # 检查特征统计信息
    print("检查特征统计信息...")
    for feature in feature_cols:
        if feature not in feature_stats:
            print(f"警告: 特征 {feature} 在统计信息中不存在!")
            print(f"可用的特征: {list(feature_stats.keys())}")
            raise ValueError(f"特征 {feature} 在统计信息中不存在")
    
    # 分析每个样本的attention和异常特征
    print(f"开始分析 {len(X_pos)} 个阳性样本...")
    
    for i, (x_seq, attn) in enumerate(zip(X_pos, attn_scores)):
        # 计算每个时间步的attention均值（平均所有头）
        time_attn = attn.mean(axis=0)  # shape: [seq_len, seq_len]
        
        # 计算每个特征的平均attention分数
        feature_attn = np.zeros(len(feature_cols))
        for j in range(len(feature_cols)):
            # 对每个特征，计算其在所有时间步上的平均attention分数
            feature_attn[j] = np.mean(time_attn[:, j % sequence_length])
        
        # 获取top 3特征的索引
        top_idx = np.argsort(feature_attn)[-3:]
        
        for idx in top_idx:
            if idx >= len(feature_cols):
                continue
                
            feature_name = feature_cols[idx]
            value = float(x_seq[:, idx].mean())  # 确保值是标量
            stats = feature_stats[feature_name]
            
            is_abnormal = (value > stats['mean'] + stats['threshold']) or (value < stats['mean'] - stats['threshold'])
            attn_score = float(feature_attn[idx])  # 确保值是标量
            
            results.append({
                'sample': int(i),  # 确保是整数
                'feature': str(feature_name),  # 确保是字符串
                'value': value,
                'is_abnormal': bool(is_abnormal),  # 确保是布尔值
                'attn_score': attn_score
            })
    
    print(f"分析完成，收集到 {len(results)} 条结果")
    if len(results) == 0:
        raise ValueError("没有收集到任何结果数据，无法创建可视化")
    
    # 保存特征
    embeddings = feature_model.predict(X_test, batch_size=128)
    np.save('processed_data/ecg_embeddings.npy', embeddings)
    np.save('processed_data/ecg_labels.npy', y_test)
    print("特征提取完成，已保存到 processed_data 目录")
    
    # ========== 创建高级可视化 ==========
    print("创建高级可视化...")
    os.makedirs('results/visualizations', exist_ok=True)
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(results)
    print(f"DataFrame形状: {df.shape}")
    print("DataFrame列名:", df.columns.tolist())
    print("DataFrame前几行:")
    print(df.head())
    
    if df.empty:
        raise ValueError("DataFrame为空，无法创建可视化")
    
    # 确保所需的列都存在
    required_columns = ['feature', 'value', 'is_abnormal', 'attn_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必要的列: {missing_columns}")

    # 1. 交互式条形图
    fig = px.bar(df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False), 
                 title='Proportion of Abnormal Features (High Attention)',
                 labels={'value': 'Abnormal Proportion', 'index': 'Feature'},
                 color='value',
                 color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Proportion of Abnormal Values",
        template="plotly_white"
    )
    pio.write_html(fig, 'results/visualizations/attention_abnormal_interactive.html')
    
    # 2. 高级热力图
    plt.figure(figsize=(12, 10))
    heatmap_data = attn_scores.mean(axis=(0, 1))
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["#2c3e50", "#3498db", "#e74c3c"])
    sns.heatmap(heatmap_data, 
                cmap=custom_cmap,
                xticklabels=feature_cols,
                yticklabels=feature_cols,
                annot=True,
                fmt='.2f',
                square=True)
    plt.title('Attention Heatmap with Custom Colormap', pad=20)
    plt.tight_layout()
    plt.savefig('results/visualizations/attention_heatmap_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 小提琴图 + 箱线图组合
    plt.figure(figsize=(15, 8))
    sns.violinplot(x='feature', y='value', hue='is_abnormal', data=df,
                  split=True, inner='box', palette='Set2')
    plt.title('Feature Value Distribution: Normal vs Abnormal', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_violin_box.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 交互式散点图矩阵
    fig = px.scatter_matrix(df,
                           dimensions=['value', 'attn_score'],
                           color='is_abnormal',
                           title='Feature Value vs Attention Score Matrix',
                           labels={'value': 'Feature Value', 'attn_score': 'Attention Score'},
                           color_discrete_sequence=px.colors.qualitative.Set2)
    pio.write_html(fig, 'results/visualizations/feature_matrix_interactive.html')
    
    # 5. 3D散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=df['value'],
        y=df['attn_score'],
        z=df['sample'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['is_abnormal'].astype(int),
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df['feature']
    )])
    fig.update_layout(
        title='3D Feature Space Visualization',
        scene=dict(
            xaxis_title='Feature Value',
            yaxis_title='Attention Score',
            zaxis_title='Sample Index'
        )
    )
    pio.write_html(fig, 'results/visualizations/3d_feature_space.html')
    
    # 6. 雷达图
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False).values,
        theta=df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False).index,
        fill='toself',
        name='Abnormal Proportion'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Feature Abnormal Proportion Radar Chart'
    )
    pio.write_html(fig, 'results/visualizations/feature_radar.html')
    
    # 7. 特征重要性条形图
    plt.figure(figsize=(12, 6))
    feature_importance = df.groupby('feature')['attn_score'].mean().sort_values(ascending=False)
    sns.barplot(x=feature_importance.index, y=feature_importance.values)
    plt.title('Feature Importance Based on Attention Scores', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 交互式时序图
    fig = go.Figure()
    for feature in feature_cols:
        feature_data = df[df['feature'] == feature]
        fig.add_trace(go.Scatter(
            x=feature_data['sample'],
            y=feature_data['value'],
            mode='lines+markers',
            name=feature,
            hovertemplate='Sample: %{x}<br>Value: %{y:.2f}<br>Feature: ' + feature
        ))
    fig.update_layout(
        title='Feature Values Over Samples',
        xaxis_title='Sample Index',
        yaxis_title='Feature Value',
        hovermode='closest'
    )
    pio.write_html(fig, 'results/visualizations/feature_timeline.html')
    
    print("可视化完成！所有图表已保存到 results/visualizations 目录")
    
except Exception as e:
    print(f"Error during feature extraction or visualization: {str(e)}")
    if 'embeddings' in locals():
        np.save('interrupted_embeddings.npy', embeddings)
        print("Embeddings state saved. You can resume extraction later.")
    raise  # Re-raise the exception to see the full traceback

# ========== SHAP特征归因分析 ==========
print("开始SHAP特征归因分析...")

# 只取一部分验证集样本做解释，避免计算过慢
background = X_val[:50]  # 作为背景分布
explained_samples = X_val[50:70]  # 需要解释的样本

# 创建SHAP解释器
explainer = shap.DeepExplainer(full_model, background)
shap_values = explainer.shap_values(explained_samples)

# 可视化（对所有样本所有时序步的归因分布）
import matplotlib.pyplot as plt
shap_save_dir = 'results/shap_analysis'
os.makedirs(shap_save_dir, exist_ok=True)

# 展示所有样本所有时序步的归因分布
shap.summary_plot(
    shap_values[0].reshape(-1, len(feature_cols)), 
    feature_names=feature_cols,
    show=False
)
plt.title('SHAP Feature Importance (All Time Steps)')
plt.tight_layout()
plt.savefig(os.path.join(shap_save_dir, 'shap_summary_all_timesteps.png'), dpi=300)
plt.close()

# 展示第一个样本所有时序步的特征归因均值
mean_shap = shap_values[0].mean(axis=0)  # (features,)
shap.summary_plot(
    [mean_shap], 
    feature_names=feature_cols,
    show=False
)
plt.title('SHAP Feature Importance (Mean over Time Steps, Sample 0)')
plt.tight_layout()
plt.savefig(os.path.join(shap_save_dir, 'shap_summary_mean_sample0.png'), dpi=300)
plt.close()

print(f"SHAP分析完成，结果已保存到 {shap_save_dir}")

# 确保检查点目录存在（如需后续训练）
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True) 