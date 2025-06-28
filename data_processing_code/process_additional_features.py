import pandas as pd
import numpy as np

# 1. 加载数据
print("步骤1: 加载数据...")
df = pd.read_excel('/Users/pursuing/Downloads/project/second_data_processing_forms/processed_dm_data.xlsx')

# 2. 目标变量转换
print("\n步骤2: 转换目标变量...")
df['target'] = df['dm_status'].apply(lambda x: 1 if x == 2 else 0)

# 3. 特征选择
print("\n步骤3: 选择特征...")
# 基础特征
base_features = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
                'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability', 
                'ecg_sequence', 'anchor_age', 'gender']

# 4. 创建新的DataFrame
print("\n步骤4: 创建新的DataFrame...")
new_df = df[base_features].copy()
new_df['target'] = df['target']
new_df['subject_id'] = df['subject_id']

# 5. 性别编码
print("\n步骤5: 处理性别编码...")
if new_df['gender'].dtype == object:
    new_df['gender'] = new_df['gender'].map({'M': 0, 'F': 1})

# 6. 添加新的特征
print("\n步骤6: 添加新特征...")

# 6.1 计算心率相关特征
new_df['heart_rate'] = 60000 / new_df['RR_Interval']  # 心率 (bpm)

# 6.2 计算QTc相关比率
new_df['QT_RR_ratio'] = new_df['QT_Interval'] / new_df['RR_Interval']
new_df['QTc_RR_ratio'] = new_df['QTc_Interval'] / new_df['RR_Interval']

# 6.3 计算波峰比率
new_df['R_P_ratio'] = new_df['R_Wave_Peak'] / new_df['P_Wave_Peak']
new_df['T_R_ratio'] = new_df['T_Wave_Peak'] / new_df['R_Wave_Peak']

# 6.4 计算年龄分组
new_df['age_group'] = pd.cut(new_df['anchor_age'],
                            bins=[0, 40, 50, 60, 70, 100],
                            labels=['<40', '40-50', '50-60', '60-70', '>70'])

# 6.5 计算心率变异性指标
new_df['HRV_CV'] = new_df['HRV_SDNN'] / new_df['RR_Interval'].mean()  # 变异系数

# 6.6 计算QT离散度
new_df['QT_dispersion'] = new_df['QT_Interval'].max() - new_df['QT_Interval'].min()

# 7. 处理缺失值
print("\n步骤7: 处理缺失值...")
# 对于数值型特征，使用中位数填充
numeric_cols = new_df.select_dtypes(include=[np.number]).columns
new_df[numeric_cols] = new_df[numeric_cols].fillna(new_df[numeric_cols].median())

# 8. 数据标准化
print("\n步骤8: 数据标准化...")
# 对数值型特征进行标准化
numeric_cols = new_df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['target', 'subject_id']]
new_df[numeric_cols] = (new_df[numeric_cols] - new_df[numeric_cols].mean()) / new_df[numeric_cols].std()

# 9. 保存处理后的数据
print("\n步骤9: 保存数据...")
new_df.to_excel('data_binary2.xlsx', index=False)

# 10. 输出数据统计信息
print("\n数据统计信息：")
print(f"总样本数: {len(new_df)}")
print(f"正样本数: {sum(new_df['target'] == 1)}")
print(f"负样本数: {sum(new_df['target'] == 0)}")
print(f"正样本比例: {sum(new_df['target'] == 1)/len(new_df):.2%}")

# 11. 显示特征列表
print("\n特征列表：")
print(new_df.columns.tolist())

# 12. 显示数据预览
print("\n数据预览：")
print(new_df.head()) 