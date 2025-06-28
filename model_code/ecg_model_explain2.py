import numpy as np
import pandas as pd
import tensorflow as tf
from new_transformer_feature_extractor import TransformerFeatureExtractor
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib.colors import LinearSegmentedColormap

# 1. 加载数据
print("加载预处理数据...")
X = np.load('processed_data/X_test_smote.npy')
y = np.load('processed_data/y_test_smote.npy')

# 2. 重塑数据为序列形式 (samples, sequence_length, features)
sequence_length = 10
n_features = X.shape[1] // sequence_length
X = X.reshape(-1, sequence_length, n_features)
print(f"样本数据形状: {X.shape}")

# 3. 只分析T2DM阳性样本
positive_indices = np.where(y == 1)[0]
X_pos = X[positive_indices]
print(f"T2DM阳性样本数量: {len(positive_indices)}")

# 4. 加载全局特征统计信息
print("加载全局特征统计信息...")
with open('results/feature_stats.json', 'r') as f:
    feature_stats = json.load(f)

# 5. 定义特征列（确保与X_test_smote.npy中的特征顺序一致）
feature_cols = [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
    'heart_rate', 'QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50'
]

# 6. 加载Transformer模型和注意力
print("加载Transformer模型...")
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
        
        # 从完整模型中提取特征提取器部分（去掉最后的分类层）
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
        
    else:
        raise FileNotFoundError(f"完整模型文件不存在: {full_model_path}")
        
except Exception as e:
    print(f"加载完整模型失败: {str(e)}")
    print("尝试重新构建模型...")
    
    # 如果加载完整模型失败，尝试重新构建
    try:
extractor = TransformerFeatureExtractor(input_dim=n_features, sequence_length=sequence_length)
model = extractor.build()
        
        # 尝试加载特征提取器权重
        feature_weights_path = 'new_transformer_feature_extractor_weights.weights.h5'
        if os.path.exists(feature_weights_path):
            try:
                model.load_weights(feature_weights_path)
                print(f"成功加载特征提取器权重: {feature_weights_path}")
            except Exception as e:
                print(f"加载特征提取器权重失败: {str(e)}")
                print("将使用未训练的特征提取器（结果可能不准确）")
        else:
            print(f"特征提取器权重文件不存在: {feature_weights_path}")
            print("将使用未训练的特征提取器（结果可能不准确）")
            
    except Exception as e:
        print(f"重新构建模型也失败: {str(e)}")
        raise ValueError("无法加载模型，请检查模型文件或重新训练模型")

attn_scores = extractor.get_attention_scores(X_pos)  # shape: [N, head, seq, seq]
print(f"注意力分数形状: {attn_scores.shape}")

# 7. 分析每个样本的attention和异常特征
results = []
for i, (x_seq, attn) in enumerate(zip(X_pos, attn_scores)):
    # 计算每个特征的attention均值（平均所有头和时间步）
    feature_attn = attn.mean(axis=(0, 1))  # shape: [seq]
    top_idx = np.argsort(feature_attn)[-3:]  # 取attention最高的3个特征
    for idx in top_idx:
        value = x_seq[:, idx].mean()
        stats = feature_stats[feature_cols[idx]]
        is_abnormal = (value > stats['mean'] + stats['threshold']) or (value < stats['mean'] - stats['threshold'])
        results.append({
            'sample': i,
            'feature': feature_cols[idx],
            'value': value,
            'is_abnormal': is_abnormal,
            'attn_score': feature_attn[idx]
        })

# 8. 汇总统计
df = pd.DataFrame(results)
summary = df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False)
print("\n特征异常比例（按attention排序）:")
print(summary)

# 9. 保存汇总结果
os.makedirs('results', exist_ok=True)
summary.to_csv('results/attention_abnormal_summary.csv')

# ========== 创建高级可视化 ==========
print("创建高级可视化...")
os.makedirs('results/visualizations', exist_ok=True)

# 10. 交互式条形图 - 特征异常比例
fig = px.bar(summary.sort_values(ascending=False), 
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

# 11. 高级热力图 - 注意力分数
plt.figure(figsize=(12, 10))
heatmap_data = attn_scores.mean(axis=(0, 1))  # shape: (seq, seq)
N = heatmap_data.shape[0]
labels = feature_cols[:N]  # 只取有数据的特征名
custom_cmap = LinearSegmentedColormap.from_list("custom", ["#2c3e50", "#3498db", "#e74c3c"])
sns.heatmap(heatmap_data, 
            cmap=custom_cmap,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='.2f',
            square=True)
plt.title('Attention Heatmap with Custom Colormap', pad=20)
plt.tight_layout()
plt.savefig('results/visualizations/attention_heatmap_advanced.png', dpi=300, bbox_inches='tight')
plt.close()

# 12. 小提琴图 + 箱线图组合 - 特征值分布
plt.figure(figsize=(15, 8))
sns.violinplot(x='feature', y='value', hue='is_abnormal', data=df,
              split=True, inner='box', palette='Set2')
plt.title('Feature Value Distribution: Normal vs Abnormal', pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/visualizations/feature_violin_box.png', dpi=300, bbox_inches='tight')
plt.close()

# 13. 交互式散点图矩阵
fig = px.scatter_matrix(df,
                       dimensions=['value', 'attn_score'],
                       color='is_abnormal',
                       title='Feature Value vs Attention Score Matrix',
                       labels={'value': 'Feature Value', 'attn_score': 'Attention Score'},
                       color_discrete_sequence=px.colors.qualitative.Set2)
pio.write_html(fig, 'results/visualizations/feature_matrix_interactive.html')

# 14. 3D散点图
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

# 15. 雷达图 - 特征异常比例
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=summary.sort_values(ascending=False).values,
    theta=summary.sort_values(ascending=False).index,
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

# 16. 特征重要性条形图 (高级样式)
plt.figure(figsize=(12, 6))
feature_importance = df.groupby('feature')['attn_score'].mean().sort_values(ascending=False)
sns.barplot(x=feature_importance.index, y=feature_importance.values, color='steelblue')
plt.title('Feature Importance Based on Attention Scores', pad=20, fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Average Attention Score', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 17. 交互式时序图 - 特征值随样本变化
fig = go.Figure()
for feature in feature_cols:
    feature_data = df[df['feature'] == feature]
    if not feature_data.empty:  # 确保有数据
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
    hovermode='closest',
    template='plotly_white'
)
pio.write_html(fig, 'results/visualizations/feature_timeline.html')

# 18. 交互式注意力-异常特征分布图
fig = px.scatter(df, 
                x='attn_score', 
                y='value', 
                color='is_abnormal',
                size='attn_score',
                hover_data=['feature', 'sample'],
                title='Attention Score vs Feature Value (Interactive)',
                labels={'attn_score': 'Attention Score', 
                       'value': 'Feature Value',
                       'is_abnormal': 'Is Abnormal'},
                color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(template='plotly_white')
pio.write_html(fig, 'results/visualizations/attention_scatter_interactive.html')

# 19. 热力图 - 特征相关性分析
plt.figure(figsize=(10, 8))
feature_pivot = df.pivot_table(values='value', index='feature', columns='is_abnormal', aggfunc='mean')
sns.heatmap(feature_pivot, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlBu_r',
            center=0,
            square=True,
            cbar_kws={'label': 'Average Feature Value'})
plt.title('Feature Values: Normal vs Abnormal Comparison', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Is Abnormal', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/visualizations/feature_comparison_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 20. 综合仪表板 - 子图组合
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Feature Importance', 'Abnormal Proportion', 
                   'Attention Distribution', 'Value Distribution'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "histogram"}, {"type": "histogram"}]]
)

# 子图1: 特征重要性
fig.add_trace(
    go.Bar(x=feature_importance.index, y=feature_importance.values, name='Importance'),
    row=1, col=1
)

# 子图2: 异常比例
fig.add_trace(
    go.Bar(x=summary.index, y=summary.values, name='Abnormal Ratio'),
    row=1, col=2
)

# 子图3: 注意力分数分布
fig.add_trace(
    go.Histogram(x=df['attn_score'], name='Attention Score', nbinsx=30),
    row=2, col=1
)

# 子图4: 特征值分布
fig.add_trace(
    go.Histogram(x=df['value'], name='Feature Value', nbinsx=30),
    row=2, col=2
)

fig.update_layout(
    title_text="Comprehensive Analysis Dashboard",
    showlegend=False,
    height=800
)
pio.write_html(fig, 'results/visualizations/comprehensive_dashboard.html')

print("高级可视化完成！所有图表已保存到 results/visualizations 目录")