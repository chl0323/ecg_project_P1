import pandas as pd
import numpy as np

# 读取筛选后的数据
print("读取数据...")
df = pd.read_excel("filtered_dm_record_list.xlsx")

# 获取所有数值型列（除了subject_id, dm_status）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['subject_id', 'dm_status']]

# 获取每列的数据类型
column_dtypes = df.dtypes

# 计算各组的统计值
print("计算统计值...")
# 计算非糖尿病组(dm_status=0)的统计值
dm0_stats = df[df['dm_status'] == 0].agg({
    'gender': lambda x: x.mode()[0],  # 众数
    **{col: 'median' for col in numeric_cols}  # 中位数
})

# 计算糖尿病组(dm_status=1)的统计值
dm1_stats = df[df['dm_status'] == 1].agg({
    'gender': lambda x: x.mode()[0],  # 众数
    **{col: 'median' for col in numeric_cols}  # 中位数
})

# 创建副本进行处理
df_processed = df.copy()

# 处理缺失值和0值
print("处理缺失值和0值...")
for index, row in df_processed.iterrows():
    subject_id = row['subject_id']
    dm_status = row['dm_status']
    
    # 获取该subject_id的所有记录
    subject_data = df[df['subject_id'] == subject_id]
    
    # 遍历每一列
    for col in df_processed.columns:
        # 检查是否为缺失值或0值
        if pd.isna(row[col]) or (col in numeric_cols and row[col] == 0):
            # 尝试使用该subject_id的有效值
            if col == 'gender':
                # 对于gender，使用众数
                valid_values = subject_data[col].dropna()
                if not valid_values.empty:
                    subject_value = valid_values.mode()[0]
                else:
                    subject_value = None
            else:
                # 对于其他列，使用非零平均值
                valid_values = subject_data[col][subject_data[col] != 0].dropna()
                if not valid_values.empty:
                    subject_value = valid_values.mean()
                else:
                    subject_value = None
            
            # 如果subject_id没有有效值，则使用对应dm_status的统计值
            if pd.isna(subject_value) or subject_value == 0:
                if dm_status == 0:
                    df_processed.at[index, col] = dm0_stats[col]
                else:
                    df_processed.at[index, col] = dm1_stats[col]
            else:
                # 根据原始数据类型进行转换
                if col in numeric_cols:
                    if column_dtypes[col] == np.int64:
                        df_processed.at[index, col] = int(round(subject_value))
                    else:
                        df_processed.at[index, col] = float(subject_value)
                else:
                    df_processed.at[index, col] = subject_value

# 保存处理后的数据
print("保存处理后的数据...")
df_processed.to_excel("processed_dm_data.xlsx", index=False)

# 打印处理结果统计
print("\n处理结果统计:")
print(f"原始数据缺失值和0值数量:")
for col in df.columns:
    missing_count = df[col].isna().sum()
    zero_count = ((df[col] == 0) & (col in numeric_cols)).sum()
    print(f"{col}: 缺失值={missing_count}, 0值={zero_count}")

print(f"\n处理后数据缺失值和0值数量:")
for col in df_processed.columns:
    missing_count = df_processed[col].isna().sum()
    zero_count = ((df_processed[col] == 0) & (col in numeric_cols)).sum()
    print(f"{col}: 缺失值={missing_count}, 0值={zero_count}")

# 保存处理统计信息
stats_df = pd.DataFrame({
    '字段': numeric_cols + ['gender'],
    '非糖尿病组中位数/众数': [dm0_stats[col] for col in numeric_cols + ['gender']],
    '糖尿病组中位数/众数': [dm1_stats[col] for col in numeric_cols + ['gender']]
})
stats_df.to_excel("missing_value_stats.xlsx", index=False)
print("\n统计信息已保存到 missing_value_stats.xlsx") 