import pandas as pd
import numpy as np

# 1. 处理糖尿病状态标记
print("步骤1: 处理糖尿病状态标记...")

# 读取诊断数据
diagnosis_df = pd.read_excel("diagnosis.xlsx")
# 读取ECG记录数据
ecg_df = pd.read_excel("new_record_list_2version.xlsx")

# 获取每列的数据类型
column_dtypes = ecg_df.dtypes

# 检查原始数据的负值
print("\n检查原始数据负值:")
numeric_cols = ecg_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['subject_id']]
for col in numeric_cols:
    neg_count = (ecg_df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: 发现 {neg_count} 个负值")

# 筛选出包含E11的ICD编码
dm_subjects = diagnosis_df[diagnosis_df['icd_code'].str.contains('E11', na=False)]['subject_id'].unique()

# 创建糖尿病状态标记
ecg_df['dm_status'] = ecg_df['subject_id'].isin(dm_subjects).astype(int)

# 保存处理后的数据
ecg_df.to_excel("dm_record_list.xlsx", index=False)
print(f"\n处理后的数据已保存到 dm_record_list.xlsx")
print(f"总记录数: {len(ecg_df)}")
print(f"糖尿病状态分布:\n{ecg_df['dm_status'].value_counts()}")

# 2. 数据筛选和统计
print("\n步骤2: 数据筛选和统计...")

# 读取处理后的数据
dm_df = pd.read_excel("dm_record_list.xlsx")

# 计算每个subject_id的记录数和RR_Interval的缺失率
subject_stats = dm_df.groupby('subject_id').agg({
    'ecg_time': 'count',  # 记录数
    'RR_Interval': lambda x: x.isna().mean()  # 缺失率
}).reset_index()

# 筛选符合条件的subject_id
valid_subjects = subject_stats[
    (subject_stats['ecg_time'] > 10) &  # 记录数>10
    (subject_stats['RR_Interval'] < 0.2)  # 缺失率<20%
]['subject_id']

# 获取符合条件的数据
filtered_df = dm_df[dm_df['subject_id'].isin(valid_subjects)].copy()

# 检查筛选后数据的负值
print("\n检查筛选后数据负值:")
for col in numeric_cols:
    neg_count = (filtered_df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: 发现 {neg_count} 个负值")
        # 将负值替换为该subject_id的正值平均值
        for subject_id in filtered_df['subject_id'].unique():
            subject_data = filtered_df[filtered_df['subject_id'] == subject_id]
            neg_mask = subject_data[col] < 0
            if neg_mask.any():
                # 获取该subject_id的正值平均值
                pos_mean = subject_data[subject_data[col] > 0][col].mean()
                if pd.notna(pos_mean):
                    # 根据原始数据类型进行转换
                    if column_dtypes[col] == np.int64:
                        filtered_df.loc[(filtered_df['subject_id'] == subject_id) & (filtered_df[col] < 0), col] = int(round(pos_mean))
                    else:
                        filtered_df.loc[(filtered_df['subject_id'] == subject_id) & (filtered_df[col] < 0), col] = float(pos_mean)

# 再次检查负值
print("\n检查处理后的负值:")
for col in numeric_cols:
    neg_count = (filtered_df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: 仍有 {neg_count} 个负值")
    else:
        print(f"{col}: 无负值")

# 统计结果
print("\n筛选结果统计:")
print(f"符合条件的总记录数: {len(filtered_df)}")
print(f"符合条件的总人数: {len(filtered_df['subject_id'].unique())}")

# 按糖尿病状态分组统计
dm_stats = filtered_df.groupby('dm_status').agg({
    'subject_id': ['count', 'nunique']
}).reset_index()

dm_stats.columns = ['dm_status', '记录数', '人数']
dm_stats['dm_status'] = dm_stats['dm_status'].map({0: '非糖尿病', 1: '糖尿病'})

print("\n按糖尿病状态统计:")
print(dm_stats)

# 保存筛选后的数据
filtered_df.to_excel("filtered_dm_record_list.xlsx", index=False)
print("\n筛选后的数据已保存到 filtered_dm_record_list.xlsx") 