import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Dict
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import os

class MMDDriftDetector:
    """
    基于最大均值差异（Maximum Mean Discrepancy, MMD）的漂移检测器
    
    Reference:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
    """
    
    def __init__(self, reference_data: np.ndarray, kernel: str = 'rbf', 
                 gamma: float = None, threshold: float = 0.1):
        """
        初始化MMD漂移检测器
        
        Args:
            reference_data: 参考数据集
            kernel: 核函数类型 ('rbf', 'linear', 'polynomial')
            gamma: RBF核的gamma参数，如果为None则使用默认值
            threshold: 漂移检测阈值
        """
        self.reference_data = reference_data
        self.kernel = kernel
        self.gamma = gamma if gamma is not None else 1.0 / reference_data.shape[1]
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
            
            # 添加数值稳定性检查
            if m <= 1 or n <= 1:
                print(f"警告：样本数量过少 (m={m}, n={n})")
                return 0.0
                
            # 计算各项的贡献
            term1 = K_XX_sum / (m * (m - 1))
            term2 = K_YY_sum / (n * (n - 1))
            term3 = 2 * K_XY_sum / (m * n)
            
            # 打印调试信息
            print(f"  MMD计算详情:")
            print(f"    term1 (K_XX): {term1:.6f}")
            print(f"    term2 (K_YY): {term2:.6f}")
            print(f"    term3 (K_XY): {term3:.6f}")
            
            mmd = term1 + term2 - term3
        else:
            # 有偏估计
            term1 = np.mean(K_XX)
            term2 = np.mean(K_YY)
            term3 = 2 * np.mean(K_XY)
            
            # 打印调试信息
            print(f"  MMD计算详情 (有偏估计):")
            print(f"    term1 (K_XX): {term1:.6f}")
            print(f"    term2 (K_YY): {term2:.6f}")
            print(f"    term3 (K_XY): {term3:.6f}")
            
            mmd = term1 + term2 - term3
        
        # 确保MMD非负，并添加小的数值稳定性常数
        mmd = max(0, mmd) + 1e-10
        
        return mmd
    
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