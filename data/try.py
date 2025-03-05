import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.tools import StandardScaler

# -------------------------------
# 1. 数据加载与预处理（保持原有结构，无聚合操作）
# -------------------------------
# ... [保持数据加载、合并、节假日处理代码不变] ...
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

    # 销售相关特征：填充缺失值避免标准化失败
    if is_train:
        # 保持原始数据顺序，按(store_nbr, family, date)分组计算特征
        df.sort_values(['store_nbr', 'family', 'date'], inplace=True)
        # 使用前向填充处理滞后特征中的NaN（例如新开业商店）
        df['sales_lag7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7).fillna(0)
        # 滚动平均同样填充缺失值
        df['rolling_avg7'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean().fillna(0)
        )
    else:
        # 测试集明确填充为0以保持结构完整
        df['sales_lag7'] = 0
        df['rolling_avg7'] = 0

    return df


train = create_features(train, is_train=True)
test = create_features(test, is_train=False)

# 编码分类变量（保持所有行）
cat_cols = ['family', 'city', 'state', 'type', 'cluster']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 定义特征列和目标列（保持所有行）
features = ['store_nbr', 'family', 'onpromotion', 'oil_price',
            'is_holiday', 'year', 'month', 'day', 'dayofweek',
            'is_month_start', 'is_month_end', 'sales_lag7', 'rolling_avg7',
            'city', 'state', 'type', 'cluster']
target = 'sales'

# 标准化前填充剩余可能的NaN（理论上不应存在）
train[features] = train[features].fillna(0)
test[features] = test[features].fillna(0)

scaler = StandardScaler()
train[features] = scaler.transform(train[features])
test[features] = scaler.transform(test[features])

# -------------------------------
# 2. 移除聚合步骤，直接使用原始数据结构
# -------------------------------
# 删除以下聚合代码：
# train_agg = ...
# test_agg = ...

# -------------------------------
# 3. 保存处理后的完整数据集
# -------------------------------
os.makedirs('./favourite', exist_ok=True)
train.to_csv('./favourite/train.csv', index=False)
test.to_csv('./favourite/test.csv', index=False)

print("CSV 文件已生成，目录：./favourite (保留全部原始数据行)")