# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
#
# # -------------------------------
# # 1. 数据加载与预处理
# # -------------------------------
# # 加载数据
# train = pd.read_csv('./store-sales-time-series-forecasting/train.csv', parse_dates=['date'])
# test = pd.read_csv('./store-sales-time-series-forecasting/test.csv', parse_dates=['date'])
# stores = pd.read_csv('./store-sales-time-series-forecasting/stores.csv')
# oil = pd.read_csv('./store-sales-time-series-forecasting/oil.csv', parse_dates=['date'])
# holidays = pd.read_csv('./store-sales-time-series-forecasting/holidays_events.csv', parse_dates=['date'])
#
# # 合并 Store 信息
# train = train.merge(stores, on='store_nbr', how='left')
# test = test.merge(stores, on='store_nbr', how='left')
#
# # 合并油价信息：修正列名
# oil = oil.rename(columns={'dcoilwtico': 'oil_price'})
# full_dates = pd.date_range(
#     start=min(train['date'].min(), oil['date'].min()),
#     end=max(train['date'].max(), oil['date'].max()),
#     freq='D'
# )
# oil_full = pd.DataFrame({'date': full_dates})
# oil_full = oil_full.merge(oil, on='date', how='left')
# oil_full['oil_price'] = oil_full['oil_price'].ffill().bfill()
#
# train = train.merge(oil_full, on='date', how='left')
# test = test.merge(oil_full, on='date', how='left')
#
# # 处理节假日：仅保留实际生效的节假日（排除 Work Day 等）
# def process_holidays(df):
#     holidays_valid = holidays[(holidays['transferred'] == False) & (holidays['type'] != 'Work Day')]
#     df['is_holiday'] = df['date'].isin(holidays_valid['date']).astype(int)
#     return df
#
# train = process_holidays(train)
# test = process_holidays(test)
#
# # 特征工程：构造日期特征、销售滞后和滚动平均特征
# def create_features(df, is_train=True):
#     # 时间特征
#     df['year'] = df['date'].dt.year
#     df['month'] = df['date'].dt.month
#     df['day'] = df['date'].dt.day
#     df['dayofweek'] = df['date'].dt.dayofweek
#     df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
#     df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
#     # 销售相关特征（仅训练集计算，测试集上 lag 特征可能缺失）
#     if is_train:
#         df.sort_values(['store_nbr', 'family', 'date'], inplace=True)
#         df['sales_lag7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
#         df['rolling_avg7'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
#             lambda x: x.rolling(7, min_periods=1).mean()
#         )
#     return df
#
# train = create_features(train, is_train=True)
# test = create_features(test, is_train=False)
#
# # 编码分类变量
# cat_cols = ['family', 'city', 'state', 'type', 'cluster']
# for col in cat_cols:
#     le = LabelEncoder()
#     train[col] = le.fit_transform(train[col])
#     test[col] = le.transform(test[col])
#
# # 定义特征列和目标列
# features = ['store_nbr', 'family', 'onpromotion', 'oil_price', 'is_holiday',
#             'year', 'month', 'day', 'dayofweek', 'is_month_start', 'is_month_end',
#             'sales_lag7', 'rolling_avg7', 'city', 'state', 'type', 'cluster']
# target = 'sales'
#
# # 标准化数值特征（这里仅对 features 做标准化，后续 Informer 的 Dataset 内部还会进行二次缩放）
# scaler = StandardScaler()
# train[features] = scaler.fit_transform(train[features])
#
# # -------------------------------
# # 2. 按日期聚合生成单一时序数据（示例：预测每日总销售额）
# # -------------------------------
# # 训练数据聚合的字典（包含销售相关特征）
# agg_dict_train = {
#     'sales': 'sum',
#     'onpromotion': 'sum',
#     'oil_price': 'mean',
#     'is_holiday': 'max',
#     'year': 'max',
#     'month': 'max',
#     'day': 'max',
#     'dayofweek': 'max',
#     'is_month_start': 'max',
#     'is_month_end': 'max',
#     'sales_lag7': 'sum',
#     'rolling_avg7': 'mean',
#     'store_nbr': 'mean',
#     'family': 'mean',
#     'city': 'mean',
#     'state': 'mean',
#     'type': 'mean',
#     'cluster': 'mean'
# }
#
# # 测试数据聚合的字典（去除不存在的销售相关特征）
# agg_dict_test = {
#     'onpromotion': 'sum',
#     'oil_price': 'mean',
#     'is_holiday': 'max',
#     'year': 'max',
#     'month': 'max',
#     'day': 'max',
#     'dayofweek': 'max',
#     'is_month_start': 'max',
#     'is_month_end': 'max',
#     'store_nbr': 'mean',
#     'family': 'mean',
#     'city': 'mean',
#     'state': 'mean',
#     'type': 'mean',
#     'cluster': 'mean'
# }
#
# # 聚合训练数据
# train_agg = train.groupby('date').agg(agg_dict_train).reset_index()
#
# # 聚合测试数据
# test_agg = test.groupby('date').agg(agg_dict_test).reset_index()
# test_agg[target] = np.nan  # 测试数据中目标列置空
#
#
#
# # 创建存储目录，并保存 CSV 文件
# os.makedirs('favourite', exist_ok=True)
# train_agg.to_csv('./favourite/train.csv', index=False)
# test_agg.to_csv('./favourite/test.csv', index=False)
#
# print("CSV 文件已生成，目录：./data/favourite")

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# 1. 数据加载与预处理
# -------------------------------
# 加载数据
train = pd.read_csv('./store-sales-time-series-forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('./store-sales-time-series-forecasting/test.csv', parse_dates=['date'])
stores = pd.read_csv('./store-sales-time-series-forecasting/stores.csv')
oil = pd.read_csv('./store-sales-time-series-forecasting/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('./store-sales-time-series-forecasting/holidays_events.csv', parse_dates=['date'])

# 合并 Store 信息
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# 合并油价信息：修正列名
oil = oil.rename(columns={'dcoilwtico': 'oil_price'})
full_dates = pd.date_range(
    start=min(train['date'].min(), oil['date'].min()),
    end=max(train['date'].max(), oil['date'].max()),
    freq='D'
)
oil_full = pd.DataFrame({'date': full_dates})
oil_full = oil_full.merge(oil, on='date', how='left')
oil_full['oil_price'] = oil_full['oil_price'].ffill().bfill()

train = train.merge(oil_full, on='date', how='left')
test = test.merge(oil_full, on='date', how='left')

# 处理节假日：仅保留实际生效的节假日（排除 Work Day 等）
def process_holidays(df):
    holidays_valid = holidays[(holidays['transferred'] == False) & (holidays['type'] != 'Work Day')]
    df['is_holiday'] = df['date'].isin(holidays_valid['date']).astype(int)
    return df

train = process_holidays(train)
test = process_holidays(test)

# 特征工程：构造日期特征、销售滞后和滚动平均特征
def create_features(df, is_train=True):
    # 时间特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    # 销售相关特征：对于训练集，计算滞后和滚动平均；对于测试集，若没有销售值，则填充默认值（例如0）
    if is_train:
        df.sort_values(['store_nbr', 'family', 'date'], inplace=True)
        df['sales_lag7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
        df['rolling_avg7'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
    else:
        # 测试集可能没有销售数据，用 0 填充
        df['sales_lag7'] = 0
        df['rolling_avg7'] = 0
    return df

train = create_features(train, is_train=True)
test = create_features(test, is_train=False)

# 编码分类变量
cat_cols = ['family', 'city', 'state', 'type', 'cluster']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 定义特征列和目标列
features = ['store_nbr', 'family', 'onpromotion', 'oil_price', 'is_holiday',
            'year', 'month', 'day', 'dayofweek', 'is_month_start', 'is_month_end',
            'sales_lag7', 'rolling_avg7', 'city', 'state', 'type', 'cluster']
target = 'sales'

# 标准化数值特征（这里只对 features 进行标准化）
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

# -------------------------------
# 2. 按多维度聚合生成时序数据
# -------------------------------
# 为了保留门店和产品系列的时序信息，采用 ['date', 'store_nbr', 'family'] 作为分组键
group_cols = ['date', 'store_nbr', 'family']

# 训练数据聚合的字典（去除 group_cols 中的列）
agg_dict_train = {
    'sales': 'sum',
    'onpromotion': 'sum',
    'oil_price': 'mean',
    'is_holiday': 'max',
    'year': 'max',
    'month': 'max',
    'day': 'max',
    'dayofweek': 'max',
    'is_month_start': 'max',
    'is_month_end': 'max',
    'sales_lag7': 'sum',
    'rolling_avg7': 'mean',
    'city': 'mean',
    'state': 'mean',
    'type': 'mean',
    'cluster': 'mean'
}

# 测试数据聚合的字典（去除销售相关特征和 group_cols 中的列）
agg_dict_test = {
    'onpromotion': 'sum',
    'oil_price': 'mean',
    'is_holiday': 'max',
    'year': 'max',
    'month': 'max',
    'day': 'max',
    'dayofweek': 'max',
    'is_month_start': 'max',
    'is_month_end': 'max',
    'city': 'mean',
    'state': 'mean',
    'type': 'mean',
    'cluster': 'mean'
}

# 聚合训练数据
train_agg = train.groupby(group_cols).agg(agg_dict_train).reset_index()

# 聚合测试数据
test_agg = test.groupby(group_cols).agg(agg_dict_test).reset_index()
test_agg[target] = np.nan  # 测试数据中目标列置空


# -------------------------------
# 3. 保存 CSV 文件
# -------------------------------
os.makedirs('./favourite', exist_ok=True)
train_agg.to_csv('./favourite/train.csv', index=False)
test_agg.to_csv('./favourite/test.csv', index=False)

print("CSV 文件已生成，目录：./favourite")
