import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

def evaluate_method(method, task_datasets):
    """
    评估持续学习方法的性能
    
    Args:
        method: 持续学习方法实例
        task_datasets: 任务数据集列表，每个元素为(X_task, y_task)元组
    
    Returns:
        results: 包含各个指标结果的字典
    """
    results = {
        'accuracy': np.zeros((len(task_datasets), len(task_datasets))),
        'f1': np.zeros((len(task_datasets), len(task_datasets))),
        'auc': np.zeros((len(task_datasets), len(task_datasets))),
        'recall': np.zeros((len(task_datasets), len(task_datasets)))
    }
    
    for task_id, (X_task, y_task) in enumerate(task_datasets):
        print(f"\n训练任务 {task_id+1}/{len(task_datasets)}")
        method.train(task_id, X_task, y_task)
        
        for eval_id, (X_eval, y_eval) in enumerate(task_datasets[:task_id+1]):
            metrics = method.evaluate(X_eval, y_eval)
            for metric_name, value in metrics.items():
                results[metric_name][task_id, eval_id] = value
                print(f"Task {task_id+1} 训练后在 Task {eval_id+1} 上 {metric_name}: {value:.4f}")
    
    return results

def plot_individual_results(results, save_dir='results'):
    """
    为每个方法绘制单独的结果图
    
    Args:
        results: 评估结果字典
        save_dir: 保存目录
    """
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
    """
    绘制所有方法的指标比较图
    
    Args:
        results: 评估结果字典
        save_dir: 保存目录
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # 计算每个方法在每个任务上的性能
        for method in methods:
            matrix = results[method][metric]
            # 只使用对角线上的值
            diagonal_scores = np.diag(matrix)
            plt.plot(range(1, len(diagonal_scores)+1), diagonal_scores, 
                    marker='o', label=method)
        
        plt.title(f'{metric.upper()} Comparison Across Tasks')
        plt.xlabel('Task')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_comparison.png')
        plt.close()

def save_results(results, save_dir='results'):
    """
    保存评估结果
    
    Args:
        results: 评估结果字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存原始结果
    for method, metrics in results.items():
        for metric_name, matrix in metrics.items():
            np.save(f'{save_dir}/{method}_{metric_name}.npy', matrix)
    
    # 计算并保存最佳方法
    best_methods = find_best_method(results)
    with open(f'{save_dir}/best_methods.txt', 'w') as f:
        f.write("最佳方法分析:\n")
        for metric, (method, score) in best_methods.items():
            f.write(f"{metric.upper()}: {method} (平均分数: {score:.4f})\n")

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

def calculate_transfer_metrics(results):
    """
    计算迁移学习指标（BWT, FWT, Forget Rate）
    
    Args:
        results: 评估结果字典
    
    Returns:
        transfer_metrics: 包含迁移指标的字典
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    
    transfer_metrics = {}
    for method in methods:
        transfer_metrics[method] = {}
        for metric in metrics:
            matrix = results[method][metric]
            
            # 计算BWT (Backward Transfer)
            bwt = np.mean([matrix[i, -1] - matrix[i, i] 
                          for i in range(matrix.shape[0] - 1)])
            
            # 计算FWT (Forward Transfer)
            fwt = np.mean([matrix[i, i-1] - matrix[i, 0] 
                          for i in range(1, matrix.shape[0])])
            
            # 计算Forget Rate
            forget_rate = np.mean([np.max(matrix[i, :i+1]) - matrix[i, -1] 
                                 for i in range(matrix.shape[0] - 1)])
            
            transfer_metrics[method][metric] = {
                'BWT': bwt,
                'FWT': fwt,
                'Forget Rate': forget_rate
            }
    
    return transfer_metrics

def plot_transfer_metrics(transfer_metrics, save_dir='results'):
    """
    绘制迁移学习指标的比较图
    
    Args:
        transfer_metrics: 迁移指标字典
        save_dir: 保存目录
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    transfer_types = ['BWT', 'FWT', 'Forget Rate']
    methods = list(transfer_metrics.keys())
    
    for transfer_type in transfer_types:
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            values = [transfer_metrics[method][metric][transfer_type] 
                     for method in methods]
            plt.plot(methods, values, marker='o', label=metric.upper())
        
        plt.title(f'{transfer_type} Comparison')
        plt.xlabel('Method')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{transfer_type}_comparison.png')
        plt.close()

def plot_forgetting_curve(results, save_dir='results'):
    """
    绘制遗忘曲线
    
    Args:
        results: 评估结果字典
        save_dir: 保存目录
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            matrix = results[method][metric]
            # 计算每个任务的最大性能和最终性能
            max_performance = np.max(matrix, axis=1)
            final_performance = matrix[:, -1]
            forgetting = max_performance - final_performance
            
            plt.plot(range(1, len(forgetting)+1), forgetting, 
                    marker='o', label=method)
        
        plt.title(f'{metric.upper()} Forgetting Curve')
        plt.xlabel('Task')
        plt.ylabel('Forgetting')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_forgetting_curve.png')
        plt.close()

def plot_learning_curve(results, save_dir='results'):
    """
    绘制学习曲线
    
    Args:
        results: 评估结果字典
        save_dir: 保存目录
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            matrix = results[method][metric]
            # 计算每个任务的平均性能
            mean_performance = np.mean(matrix, axis=1)
            
            plt.plot(range(1, len(mean_performance)+1), mean_performance, 
                    marker='o', label=method)
        
        plt.title(f'{metric.upper()} Learning Curve')
        plt.xlabel('Task')
        plt.ylabel('Average Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_learning_curve.png')
        plt.close()