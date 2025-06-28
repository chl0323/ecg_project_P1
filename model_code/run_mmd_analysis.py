import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
import sys
from pathlib import Path
from typing import List, Union, Optional, Dict

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

class MMDDriftDetector:
    """
    基于最大均值差异（Maximum Mean Discrepancy, MMD）的漂移检测器
    
    Reference:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
    """
    
    def __init__(self, reference_data: np.ndarray, kernel: str = 'rbf', 
                 gamma: float = 1.0, threshold: float = 0.1):
        """
        初始化MMD漂移检测器
        
        Args:
            reference_data: 参考数据集
            kernel: 核函数类型 ('rbf', 'linear', 'polynomial')
            gamma: RBF核的gamma参数
            threshold: 漂移检测阈值
        """
        self.reference_data = reference_data
        self.kernel = kernel
        self.gamma = gamma
        self.threshold = threshold
        self.mmd_history = []
        self.drift_points = []
        self.warning_points = []
        
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float = None) -> np.ndarray:
        """计算RBF核矩阵"""
        if gamma is None:
            gamma = self.gamma
        
        # 计算欧氏距离的平方
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        dist_sq = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        
        return np.exp(-gamma * dist_sq)
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """计算线性核矩阵"""
        return np.dot(X, Y.T)
    
    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray, 
                          degree: int = 3, coef0: float = 1.0) -> np.ndarray:
        """计算多项式核矩阵"""
        return (np.dot(X, Y.T) + coef0) ** degree
    
    def _compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """根据核类型计算核矩阵"""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        elif self.kernel == 'linear':
            return self._linear_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X, Y)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
    
    def compute_mmd(self, X: np.ndarray, Y: np.ndarray, unbiased: bool = True) -> float:
        """
        计算两个数据集之间的MMD
        
        Args:
            X: 第一个数据集
            Y: 第二个数据集
            unbiased: 是否使用无偏估计
            
        Returns:
            MMD值
        """
        m, n = len(X), len(Y)
        
        # 计算核矩阵
        K_XX = self._compute_kernel_matrix(X, X)
        K_YY = self._compute_kernel_matrix(Y, Y)
        K_XY = self._compute_kernel_matrix(X, Y)
        
        if unbiased:
            # 无偏估计（去除对角线元素）
            K_XX_sum = np.sum(K_XX) - np.trace(K_XX)
            K_YY_sum = np.sum(K_YY) - np.trace(K_YY)
            K_XY_sum = np.sum(K_XY)
            
            mmd = (K_XX_sum / (m * (m - 1)) + 
                   K_YY_sum / (n * (n - 1)) - 
                   2 * K_XY_sum / (m * n))
        else:
            # 有偏估计
            mmd = (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))
        
        return max(0, mmd)  # 确保MMD非负
    
    def update(self, new_data: np.ndarray) -> Dict:
        """
        更新检测器并检测漂移
        
        Args:
            new_data: 新的数据批次
            
        Returns:
            检测结果字典
        """
        # 计算与参考数据的MMD
        mmd_value = self.compute_mmd(self.reference_data, new_data)
        self.mmd_history.append(mmd_value)
        
        # 检测漂移
        drift_detected = mmd_value > self.threshold
        warning_detected = mmd_value > self.threshold * 0.8  # 警告阈值为80%
        
        if drift_detected:
            self.drift_points.append(len(self.mmd_history) - 1)
        
        if warning_detected and not drift_detected:
            self.warning_points.append(len(self.mmd_history) - 1)
        
        return {
            'mmd': mmd_value,
            'drift_detected': drift_detected,
            'warning_detected': warning_detected,
            'threshold': self.threshold
        }
    
    def plot_mmd_history(self, save_path: Optional[str] = None):
        """绘制MMD历史"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.mmd_history, label='MMD', color='blue', linewidth=2)
        plt.axhline(y=self.threshold, color='red', linestyle='--', 
                   label=f'Drift Threshold ({self.threshold})')
        plt.axhline(y=self.threshold * 0.8, color='orange', linestyle='--', 
                   label=f'Warning Threshold ({self.threshold * 0.8:.3f})')
        
        # 标记漂移点
        if self.drift_points:
            plt.scatter(self.drift_points, [self.mmd_history[i] for i in self.drift_points], 
                       color='red', label='Drift Detected', marker='*', s=100)
        
        # 标记警告点
        if self.warning_points:
            plt.scatter(self.warning_points, [self.mmd_history[i] for i in self.warning_points], 
                       color='orange', label='Warning', marker='^', s=60)
        
        plt.title('MMD-based Drift Detection History')
        plt.xlabel('Time Step')
        plt.ylabel('MMD Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def analyze_embedding_drift(embeddings_list: list, labels_list: list, 
                          task_names: list = None, kernel: str = 'rbf'):
    """
    分析嵌入空间的漂移
    
    Args:
        embeddings_list: 每个任务的嵌入列表
        labels_list: 每个任务的标签列表
        task_names: 任务名称列表
        kernel: MMD计算使用的核函数类型
    """
    # 创建保存目录
    save_dir = 'results/drift_detection'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    
    if task_names is None:
        task_names = [f'Task_{i+1}' for i in range(len(embeddings_list))]
    
    # 初始化结果字典
    results = {
        'mmd_matrix': np.zeros((len(task_names), len(task_names))),
        'class_wise_mmd': {},
        'statistics': {}
    }
    
    # 计算每个任务的统计信息
    print("\n计算任务统计信息...")
    for i, (emb, lab) in enumerate(zip(embeddings_list, labels_list)):
        task_stats = {
            'mean': np.mean(emb, axis=0),
            'std': np.std(emb, axis=0),
            'min': np.min(emb, axis=0),
            'max': np.max(emb, axis=0),
            'class_distribution': {
                '0': np.sum(lab == 0),
                '1': np.sum(lab == 1)
            }
        }
        results['statistics'][task_names[i]] = task_stats
        
        print(f"\n{task_names[i]} 统计信息:")
        print(f"  均值范围: [{task_stats['mean'].min():.4f}, {task_stats['mean'].max():.4f}]")
        print(f"  标准差范围: [{task_stats['std'].min():.4f}, {task_stats['std'].max():.4f}]")
        print(f"  类别分布: 0={task_stats['class_distribution']['0']}, 1={task_stats['class_distribution']['1']}")
    
    # 计算任务间的MMD
    print("\n计算任务间的MMD...")
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            print(f"\n计算 {task_names[i]} vs {task_names[j]} 的MMD:")
            print(f"  {task_names[i]} 数据范围: [{embeddings_list[i].min():.4f}, {embeddings_list[i].max():.4f}]")
            print(f"  {task_names[j]} 数据范围: [{embeddings_list[j].min():.4f}, {embeddings_list[j].max():.4f}]")
            
            # 计算数据维度的中位数距离作为gamma的参考
            X = embeddings_list[i]
            Y = embeddings_list[j]
            X_norm = np.sum(X**2, axis=1, keepdims=True)
            Y_norm = np.sum(Y**2, axis=1, keepdims=True)
            dist_sq = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
            median_dist = np.median(dist_sq[dist_sq > 0])
            gamma = 1.0 / median_dist if median_dist > 0 else 1.0
            
            print(f"  使用gamma = {gamma:.4f}")
            detector = MMDDriftDetector(embeddings_list[i], kernel=kernel, gamma=gamma)
            mmd_value = detector.compute_mmd(embeddings_list[i], embeddings_list[j], unbiased=False)
            
            # 打印核矩阵的统计信息
            K_XX = detector._compute_kernel_matrix(embeddings_list[i], embeddings_list[i])
            K_YY = detector._compute_kernel_matrix(embeddings_list[j], embeddings_list[j])
            K_XY = detector._compute_kernel_matrix(embeddings_list[i], embeddings_list[j])
            
            print(f"  K_XX 范围: [{K_XX.min():.4f}, {K_XX.max():.4f}], 均值: {K_XX.mean():.4f}")
            print(f"  K_YY 范围: [{K_YY.min():.4f}, {K_YY.max():.4f}], 均值: {K_YY.mean():.4f}")
            print(f"  K_XY 范围: [{K_XY.min():.4f}, {K_XY.max():.4f}], 均值: {K_XY.mean():.4f}")
            print(f"  MMD值: {mmd_value:.4f}")
            
            results['mmd_matrix'][i, j] = mmd_value
            results['mmd_matrix'][j, i] = mmd_value
    
    # 可视化MMD矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['mmd_matrix'], annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=task_names, yticklabels=task_names)
    plt.title('Task-wise MMD Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', 'mmd_matrix.png'))
    plt.close()
    
    # 类级别的MMD分析
    print("\n执行类级别的MMD分析...")
    unique_labels = np.unique(np.concatenate(labels_list))
    for label in unique_labels:
        class_results = np.zeros((len(task_names), len(task_names)))
        for i in range(len(task_names)):
            for j in range(i+1, len(task_names)):
                # 获取当前类的样本
                mask_i = labels_list[i] == label
                mask_j = labels_list[j] == label
                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    detector = MMDDriftDetector(embeddings_list[i][mask_i], kernel=kernel)
                    mmd_value = detector.compute_mmd(embeddings_list[i][mask_i], 
                                                   embeddings_list[j][mask_j],
                                                   unbiased=False)
                    class_results[i, j] = mmd_value
                    class_results[j, i] = mmd_value
        
        results['class_wise_mmd'][f'class_{int(label)}'] = [[float(val) for val in row] 
                                                           for row in class_results]
        
        # 可视化类级别的MMD矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(class_results, annot=True, fmt='.4f', cmap='YlOrRd',
                    xticklabels=task_names, yticklabels=task_names)
        plt.title(f'Class {int(label)} MMD Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plots', f'class_{int(label)}_mmd_matrix.png'))
        plt.close()
    
    # 时序MMD分析
    if len(task_names) > 1:
        print("\n执行时序MMD分析...")
        temporal_mmd = []
        for i in range(len(task_names)-1):
            detector = MMDDriftDetector(embeddings_list[i], kernel=kernel)
            mmd_value = detector.compute_mmd(embeddings_list[i], embeddings_list[i+1])
            temporal_mmd.append(mmd_value)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(temporal_mmd)), temporal_mmd, 'b-o')
        plt.title('Temporal MMD Analysis')
        plt.xlabel('Time Step')
        plt.ylabel('MMD Value')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'plots', 'temporal_mmd.png'))
        plt.close()
        
        results['temporal_mmd'] = temporal_mmd
    
    # 创建交互式MMD矩阵可视化
    fig = go.Figure(data=go.Heatmap(
        z=results['mmd_matrix'],
        x=task_names,
        y=task_names,
        colorscale='YlOrRd',
        text=[[f'{val:.4f}' for val in row] for row in results['mmd_matrix']],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Interactive MMD Matrix',
        xaxis_title='Tasks',
        yaxis_title='Tasks'
    )
    
    pio.write_html(fig, os.path.join(save_dir, 'plots', 'interactive_mmd_matrix.html'))
    
    # 可视化数据分布
    print("\n生成数据分布可视化...")
    plt.figure(figsize=(15, 10))
    
    # 1. 均值分布对比
    plt.subplot(2, 2, 1)
    means = [stats['mean'] for stats in results['statistics'].values()]
    plt.boxplot(means, tick_labels=task_names)
    plt.title('Mean Distribution Across Tasks')
    plt.ylabel('Mean Value')
    
    # 2. 标准差分布对比
    plt.subplot(2, 2, 2)
    stds = [stats['std'] for stats in results['statistics'].values()]
    plt.boxplot(stds, tick_labels=task_names)
    plt.title('Standard Deviation Distribution Across Tasks')
    plt.ylabel('Standard Deviation')
    
    # 3. 类别分布对比
    plt.subplot(2, 2, 3)
    class_dist = np.array([[stats['class_distribution']['0'], stats['class_distribution']['1']] 
                          for stats in results['statistics'].values()])
    x = np.arange(len(task_names))
    width = 0.35
    plt.bar(x - width/2, class_dist[:, 0], width, label='Class 0')
    plt.bar(x + width/2, class_dist[:, 1], width, label='Class 1')
    plt.title('Class Distribution Across Tasks')
    plt.xticks(x, task_names)
    plt.ylabel('Number of Samples')
    plt.legend()
    
    # 4. MMD矩阵热图
    plt.subplot(2, 2, 4)
    sns.heatmap(results['mmd_matrix'], annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=task_names, yticklabels=task_names)
    plt.title('MMD Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', 'distribution_analysis.png'))
    plt.close()
    
    # 保存统计信息
    with open(os.path.join(save_dir, 'metrics', 'statistics.json'), 'w') as f:
        json.dump({
            'task_statistics': {
                name: {
                    'mean_range': [float(stats['mean'].min()), float(stats['mean'].max())],
                    'std_range': [float(stats['std'].min()), float(stats['std'].max())],
                    'class_distribution': {
                        '0': int(stats['class_distribution']['0']),
                        '1': int(stats['class_distribution']['1'])
                    }
                }
                for name, stats in results['statistics'].items()
            },
            'mmd_matrix': [[float(val) for val in row] for row in results['mmd_matrix']]
        }, f, indent=4)
    
    print(f"\n分析完成。结果已保存到: {save_dir}")

def main():
    # 加载数据
    print("加载数据...")
    try:
        embeddings = np.load('processed_data/ecg_embeddings.npy')
        labels = np.load('processed_data/ecg_labels.npy')
        print(f"嵌入数据形状: {embeddings.shape}")
        print(f"标签数据形状: {labels.shape}")
        print(f"嵌入数据范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"唯一标签值: {np.unique(labels)}")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return
    
    # 将数据分成三个任务
    task_size = len(embeddings) // 3
    embeddings_list = [
        embeddings[:task_size],
        embeddings[task_size:2*task_size],
        embeddings[2*task_size:]
    ]
    labels_list = [
        labels[:task_size],
        labels[task_size:2*task_size],
        labels[2*task_size:]
    ]
    
    print("\n任务数据统计:")
    for i, (emb, lab) in enumerate(zip(embeddings_list, labels_list)):
        print(f"Task{i+1}:")
        print(f"  样本数: {len(emb)}")
        print(f"  嵌入范围: [{emb.min():.4f}, {emb.max():.4f}]")
        print(f"  唯一标签: {np.unique(lab)}")
    
    # 执行分析
    analyze_embedding_drift(
        embeddings_list=embeddings_list,
        labels_list=labels_list,
        task_names=['Task1', 'Task2', 'Task3'],
        kernel='rbf'
    )

if __name__ == '__main__':
    main() 