import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import json

class StatisticalAnalyzer:
    """
    统计显著性分析器
    用于评估模型性能差异的统计显著性
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        初始化统计分析器
        
        Args:
            alpha: 显著性水平，默认0.05
        """
        self.alpha = alpha
        self.results = {}
        
    def t_test(self, group1: np.ndarray, group2: np.ndarray, 
               paired: bool = False, alternative: str = 'two-sided') -> Dict:
        """
        执行t检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            paired: 是否为配对t检验
            alternative: 假设检验类型 ('two-sided', 'less', 'greater')
            
        Returns:
            包含检验结果的字典
        """
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
            test_type = "Paired t-test"
        else:
            statistic, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
            test_type = "Independent t-test"
            
        result = {
            'test_type': test_type,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': p_value < self.alpha,
            'effect_size': self._calculate_cohens_d(group1, group2),
            'group1_stats': {
                'mean': float(np.mean(group1)),
                'std': float(np.std(group1, ddof=1)),
                'n': len(group1)
            },
            'group2_stats': {
                'mean': float(np.mean(group2)),
                'std': float(np.std(group2, ddof=1)),
                'n': len(group2)
            }
        }
        
        return result
    
    def wilcoxon_test(self, group1: np.ndarray, group2: np.ndarray = None,
                      alternative: str = 'two-sided') -> Dict:
        """
        执行Wilcoxon检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据（如果为None，则执行单样本Wilcoxon检验）
            alternative: 假设检验类型
            
        Returns:
            包含检验结果的字典
        """
        if group2 is None:
            # 单样本Wilcoxon检验
            statistic, p_value = stats.wilcoxon(group1, alternative=alternative)
            test_type = "One-sample Wilcoxon signed-rank test"
        else:
            # 配对Wilcoxon检验
            statistic, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
            test_type = "Wilcoxon signed-rank test"
            
        result = {
            'test_type': test_type,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': p_value < self.alpha,
            'group1_stats': {
                'median': float(np.median(group1)),
                'iqr': float(np.percentile(group1, 75) - np.percentile(group1, 25)),
                'n': len(group1)
            }
        }
        
        if group2 is not None:
            result['group2_stats'] = {
                'median': float(np.median(group2)),
                'iqr': float(np.percentile(group2, 75) - np.percentile(group2, 25)),
                'n': len(group2)
            }
            
        return result
    
    def mann_whitney_u_test(self, group1: np.ndarray, group2: np.ndarray,
                           alternative: str = 'two-sided') -> Dict:
        """
        执行Mann-Whitney U检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            alternative: 假设检验类型
            
        Returns:
            包含检验结果的字典
        """
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        result = {
            'test_type': 'Mann-Whitney U test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': p_value < self.alpha,
            'effect_size': self._calculate_rank_biserial_correlation(group1, group2),
            'group1_stats': {
                'median': float(np.median(group1)),
                'iqr': float(np.percentile(group1, 75) - np.percentile(group1, 25)),
                'n': len(group1)
            },
            'group2_stats': {
                'median': float(np.median(group2)),
                'iqr': float(np.percentile(group2, 75) - np.percentile(group2, 25)),
                'n': len(group2)
            }
        }
        
        return result
    
    def anova_test(self, *groups: np.ndarray) -> Dict:
        """
        执行单因素方差分析（ANOVA）
        
        Args:
            *groups: 多个组的数据
            
        Returns:
            包含检验结果的字典
        """
        statistic, p_value = stats.f_oneway(*groups)
        
        result = {
            'test_type': 'One-way ANOVA',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': p_value < self.alpha,
            'num_groups': len(groups),
            'group_stats': []
        }
        
        for i, group in enumerate(groups):
            result['group_stats'].append({
                'group': i + 1,
                'mean': float(np.mean(group)),
                'std': float(np.std(group, ddof=1)),
                'n': len(group)
            })
            
        return result
    
    def kruskal_wallis_test(self, *groups: np.ndarray) -> Dict:
        """
        执行Kruskal-Wallis检验
        
        Args:
            *groups: 多个组的数据
            
        Returns:
            包含检验结果的字典
        """
        statistic, p_value = stats.kruskal(*groups)
        
        result = {
            'test_type': 'Kruskal-Wallis test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': p_value < self.alpha,
            'num_groups': len(groups),
            'group_stats': []
        }
        
        for i, group in enumerate(groups):
            result['group_stats'].append({
                'group': i + 1,
                'median': float(np.median(group)),
                'iqr': float(np.percentile(group, 75) - np.percentile(group, 25)),
                'n': len(group)
            })
            
        return result
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)
    
    def _calculate_rank_biserial_correlation(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """计算等级双列相关系数"""
        n1, n2 = len(group1), len(group2)
        u1, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return float(1 - (2 * u1) / (n1 * n2))
    
    def compare_model_performance(self, performance_data: Dict[str, List[float]], 
                                test_type: str = 'auto') -> Dict:
        """
        比较多个模型的性能
        
        Args:
            performance_data: 格式为 {'model_name': [performance_scores]}
            test_type: 检验类型 ('auto', 't_test', 'mann_whitney', 'anova', 'kruskal_wallis')
            
        Returns:
            包含所有比较结果的字典
        """
        model_names = list(performance_data.keys())
        model_scores = list(performance_data.values())
        
        # 自动选择检验类型
        if test_type == 'auto':
            if len(model_names) == 2:
                # 检查正态性
                normal1 = self._test_normality(model_scores[0])
                normal2 = self._test_normality(model_scores[1])
                if normal1 and normal2:
                    test_type = 't_test'
                else:
                    test_type = 'mann_whitney'
            else:
                # 多组比较，检查所有组的正态性
                all_normal = all(self._test_normality(scores) for scores in model_scores)
                if all_normal:
                    test_type = 'anova'
                else:
                    test_type = 'kruskal_wallis'
        
        results = {
            'comparison_type': test_type,
            'models': model_names,
            'pairwise_comparisons': [],
            'overall_test': None
        }
        
        # 执行相应的检验
        if test_type == 't_test' and len(model_names) == 2:
            result = self.t_test(model_scores[0], model_scores[1])
            results['overall_test'] = result
            
        elif test_type == 'mann_whitney' and len(model_names) == 2:
            result = self.mann_whitney_u_test(model_scores[0], model_scores[1])
            results['overall_test'] = result
            
        elif test_type == 'anova':
            result = self.anova_test(*model_scores)
            results['overall_test'] = result
            
        elif test_type == 'kruskal_wallis':
            result = self.kruskal_wallis_test(*model_scores)
            results['overall_test'] = result
        
        # 执行成对比较
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                if test_type in ['t_test', 'anova']:
                    comparison = self.t_test(model_scores[i], model_scores[j])
                else:
                    comparison = self.mann_whitney_u_test(model_scores[i], model_scores[j])
                
                # 确保所有值都是 Python 原生类型
                comparison = {k: bool(v) if isinstance(v, np.bool_) else v 
                            for k, v in comparison.items()}
                comparison['model1'] = model_names[i]
                comparison['model2'] = model_names[j]
                results['pairwise_comparisons'].append(comparison)
        
        # 确保 overall_test 中的值也是 Python 原生类型
        if results['overall_test']:
            results['overall_test'] = {k: bool(v) if isinstance(v, np.bool_) else v 
                                     for k, v in results['overall_test'].items()}
        
        return results
    
    def _test_normality(self, data: np.ndarray, test: str = 'shapiro') -> bool:
        """
        检验数据的正态性
        
        Args:
            data: 待检验数据
            test: 检验方法 ('shapiro', 'anderson', 'kolmogorov')
            
        Returns:
            是否符合正态分布
        """
        if test == 'shapiro':
            _, p_value = stats.shapiro(data)
        elif test == 'anderson':
            result = stats.anderson(data, dist='norm')
            # 使用5%显著性水平的临界值
            p_value = 0.05 if result.statistic > result.critical_values[2] else 0.1
        else:  # kolmogorov
            _, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        
        return p_value > self.alpha
    
    def generate_statistical_report(self, results: Dict, save_path: str = None) -> str:
        """
        生成统计分析报告
        
        Args:
            results: 统计检验结果
            save_path: 保存路径
            
        Returns:
            报告内容
        """
        report = []
        report.append("=" * 60)
        report.append("统计显著性分析报告")
        report.append("=" * 60)
        report.append("")
        
        if 'overall_test' in results and results['overall_test']:
            test = results['overall_test']
            report.append(f"整体检验: {test['test_type']}")
            report.append(f"检验统计量: {test['statistic']:.4f}")
            report.append(f"p值: {test['p_value']:.6f}")
            report.append(f"显著性水平: {test['alpha']}")
            report.append(f"是否显著: {'是' if test['significant'] else '否'}")
            report.append("")
        
        if 'pairwise_comparisons' in results:
            report.append("成对比较结果:")
            report.append("-" * 40)
            
            for comparison in results['pairwise_comparisons']:
                report.append(f"{comparison['model1']} vs {comparison['model2']}")
                report.append(f"  检验类型: {comparison['test_type']}")
                report.append(f"  p值: {comparison['p_value']:.6f}")
                report.append(f"  显著性: {'是' if comparison['significant'] else '否'}")
                
                if 'effect_size' in comparison:
                    report.append(f"  效应量: {comparison['effect_size']:.4f}")
                
                report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def visualize_comparison(self, performance_data: Dict[str, List[float]], 
                           save_dir: str = 'results/statistical_analysis'):
        """
        可视化模型性能比较
        
        Args:
            performance_data: 性能数据
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 箱线图比较
        plt.figure(figsize=(12, 8))
        data_for_plot = []
        labels_for_plot = []
        
        for model_name, scores in performance_data.items():
            data_for_plot.extend(scores)
            labels_for_plot.extend([model_name] * len(scores))
        
        df = pd.DataFrame({'Model': labels_for_plot, 'Performance': data_for_plot})
        
        sns.boxplot(data=df, x='Model', y='Performance')
        plt.title('Model Performance Comparison (Box Plot)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 小提琴图
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=df, x='Model', y='Performance')
        plt.title('Model Performance Comparison (Violin Plot)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_violinplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 交互式箱线图
        fig = px.box(df, x='Model', y='Performance', 
                     title='Interactive Model Performance Comparison')
        fig.update_layout(template='plotly_white')
        pio.write_html(fig, f'{save_dir}/performance_interactive_boxplot.html')
        
        # 4. 统计汇总表
        summary_stats = []
        for model_name, scores in performance_data.items():
            summary_stats.append({
                'Model': model_name,
                'Mean': np.mean(scores),
                'Std': np.std(scores, ddof=1),
                'Median': np.median(scores),
                'Min': np.min(scores),
                'Max': np.max(scores),
                'N': len(scores)
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f'{save_dir}/performance_summary.csv', index=False)
        
        print(f"可视化结果已保存到: {save_dir}")


def analyze_model_results(results_file: str = None, performance_data: Dict = None,
                         save_dir: str = 'results/statistical_analysis'):
    """
    分析模型结果的统计显著性
    
    Args:
        results_file: 结果文件路径（JSON格式）
        performance_data: 性能数据字典
        save_dir: 保存目录
    """
    # 创建分析器
    analyzer = StatisticalAnalyzer()
    
    # 加载数据
    if results_file and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            performance_data = json.load(f)
    
    if not performance_data:
        print("错误: 未提供性能数据")
        return
    
    # 执行统计分析
    print("开始统计显著性分析...")
    comparison_results = analyzer.compare_model_performance(performance_data)
    
    # 生成报告
    report = analyzer.generate_statistical_report(
        comparison_results, 
        save_path=f'{save_dir}/statistical_report.txt'
    )
    print(report)
    
    # 生成可视化
    analyzer.visualize_comparison(performance_data, save_dir)
    
    # 保存详细结果
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/detailed_results.json', 'w') as f:
        # 确保所有数据都是 JSON 可序列化的
        def convert_to_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(comparison_results)
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    print(f"分析完成！结果已保存到: {save_dir}")


if __name__ == "__main__":
    # 示例用法
    example_data = {
        'Transformer': [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.87, 0.85, 0.86, 0.88],
        'LSTM': [0.82, 0.83, 0.84, 0.81, 0.85, 0.83, 0.82, 0.84, 0.83, 0.82],
        'CNN': [0.78, 0.79, 0.80, 0.77, 0.81, 0.79, 0.78, 0.80, 0.79, 0.78]
    }
    
    analyze_model_results(performance_data=example_data) 