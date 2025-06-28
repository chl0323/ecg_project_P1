import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# 设置 Google Drive 路径
drive_path = '/content/drive/MyDrive/ecg_colab_final'
if os.path.exists(drive_path):
    os.chdir(drive_path)
    sys.path.append(os.path.join(drive_path, 'CL_code'))

from base import MLP
from ewc import EWC
from replay import Replay
from ranpac import RanPAC
from lwf import LwF
from gem import GEM
from icarl import iCaRL
from evaluation import evaluate_method, save_results
from statistical_analysis import StatisticalAnalyzer


def plot_individual_results(results, save_dir='results'):
    """为每个方法绘制单独的结果图"""
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())

    for method in methods:
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            matrix = results[method][metric]

            # 绘制热力图
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title(f'{method} - {metric.upper()}')
            plt.xlabel('Task')
            plt.ylabel('Task')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/{method}_results.png')
        plt.close()


def plot_comparison_metrics(results, save_dir='results'):
    """绘制所有方法的指标比较图"""
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())

    for metric in metrics:
        plt.figure(figsize=(12, 6))

        # 计算每个方法在每个任务上的平均性能
        task_means = []
        for method in methods:
            matrix = results[method][metric]
            means = np.mean(matrix, axis=1)  # 每个任务的平均性能
            task_means.append(means)

        # 绘制折线图
        for i, method in enumerate(methods):
            plt.plot(range(1, len(task_means[i]) + 1), task_means[i],
                     marker='o', label=method)

        plt.title(f'{metric.upper()} Comparison Across Tasks')
        plt.xlabel('Task')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_comparison.png')
        plt.close()


def find_best_method(results):
    """
    找出每个指标上表现最好的方法

    Args:
        results: 评估结果字典

    Returns:
        best_methods: 包含每个指标最佳方法的字典
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())

    best_methods = {}
    for metric in metrics:
        best_score = -1
        best_method = None

        for method in methods:
            matrix = results[method][metric]
            # 计算每个任务在当前任务上的性能
            current_task_scores = []
            for i in range(matrix.shape[0]):
                current_task_scores.append(matrix[i, i])

            # 计算平均性能
            mean_score = np.mean(current_task_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_method = method

        best_methods[metric] = (best_method, best_score)

    return best_methods


def main():
    # 加载数据
    embeddings = np.load('processed_data/ecg_embeddings.npy')
    labels = np.load('processed_data/ecg_labels.npy')

    # 准备任务
    N_TASKS = 5
    indices = np.arange(embeddings.shape[0])
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, N_TASKS)

    task_datasets = []
    for idx in split_indices:
        X_task = embeddings[idx]
        y_task = labels[idx]
        task_datasets.append((X_task, y_task))

    # 初始化模型和方法
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim=embeddings.shape[1]).to(device)

    methods = {
        'EWC': EWC(model, device, lambda_=5000),
        'Replay': Replay(model, device, memory_size=1000),
        'RanPAC': RanPAC(model, device, projection_dim=128, lambda_=1000),
        'LwF': LwF(model, device, temperature=2.0, lambda_=1.0),
        'GEM': GEM(model, device, memory_size=1000),
        'iCaRL': iCaRL(model, device, memory_size=1000)
    }

    # 评估所有方法
    results = {}
    for method_name, method in methods.items():
        print(f"\n评估 {method_name}...")
        results[method_name] = evaluate_method(method, task_datasets)

    # 保存结果
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    save_results(results, save_dir)

    # 绘制单独的结果图
    plot_individual_results(results, save_dir)

    # 绘制比较图
    plot_comparison_metrics(results, save_dir)

    # 找出最佳方法
    best_methods = find_best_method(results)

    # 打印最佳方法
    print("\n最佳方法分析:")
    for metric, (method, score) in best_methods.items():
        print(f"{metric.upper()}: {method} (平均分数: {score:.4f})")

    # 保存最佳方法结果
    with open(f'{save_dir}/best_methods.txt', 'w') as f:
        f.write("最佳方法分析:\n")
        for metric, (method, score) in best_methods.items():
            f.write(f"{metric.upper()}: {method} (平均分数: {score:.4f})\n")

    # === 集成统计显著性分析 ===
    stat_save_dir = os.path.join(save_dir, 'statistical_analysis')
    os.makedirs(stat_save_dir, exist_ok=True)
    analyzer = StatisticalAnalyzer()
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    for metric in metrics:
        # 收集每个方法的对角线（即每个任务的最终指标）
        performance_data = {method: np.diag(results[method][metric]).tolist() for method in results}
        # 统计检验
        stat_results = analyzer.compare_model_performance(performance_data)
        # 生成报告
        report_path = os.path.join(stat_save_dir, f'{metric}_stat_report.txt')
        analyzer.generate_statistical_report(stat_results, save_path=report_path)
        # 可视化
        analyzer.visualize_comparison(performance_data, save_dir=stat_save_dir)
    print(f"\n统计显著性分析已完成，结果保存在: {stat_save_dir}")


if __name__ == '__main__':
    main()