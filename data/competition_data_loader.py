import pandas as pd
import numpy as np
from sklearn.preprocessing  import LabelEncoder, StandardScaler

# 加载所有数据
train = pd.read_csv('./store-sales-time-series-forecasting/train.csv',  parse_dates=['date'])
test = pd.read_csv('./store-sales-time-series-forecasting/test.csv',  parse_dates=['date'])
stores = pd.read_csv('./store-sales-time-series-forecasting/stores.csv')
oil = pd.read_csv('./store-sales-time-series-forecasting/oil.csv',  parse_dates=['date'])
holidays = pd.read_csv('./store-sales-time-series-forecasting/holidays_events.csv',  parse_dates=['date'])

# 合并Store信息
train = train.merge(stores,  on='store_nbr', how='left')
test = test.merge(stores,  on='store_nbr', how='left')

# 合并油价信息
# 修正列名修改的语法错误
oil = oil.rename(columns={'dcoilwtico':  'oil_price'})

# 创建完整日期范围
full_dates = pd.date_range(
    start=min(train['date'].min(), oil['date'].min()),
    end=max(train['date'].max(), oil['date'].max()),
    freq='D'
)

# 重构完整油价序列
oil_full = pd.DataFrame({'date': full_dates})
oil_full = oil_full.merge(oil,  on='date', how='left')
oil_full['oil_price'] = oil_full['oil_price'].ffill().bfill()  # 双向填充

# 重新合并数据
train = train.merge(oil_full,  on='date', how='left')
test = test.merge(oil_full,  on='date', how='left')

# 处理节假日
def process_holidays(df):
    # 只保留实际生效的节假日
    holidays_valid = holidays[(holidays['transferred'] == False) & (holidays['type'] != 'Work Day')]
    df['is_holiday'] = df['date'].isin(holidays_valid['date']).astype(int)
    return df

train = process_holidays(train)
test = process_holidays(test)

# 特征工程
def create_features(df, is_train=True):
    # 时间特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # 仅训练数据生成销售相关特征
    if is_train:
        df.sort_values(['store_nbr',  'family', 'date'], inplace=True)
        df['sales_lag7'] = df.groupby(['store_nbr',  'family'])['sales'].shift(7)
        df['rolling_avg7'] = df.groupby(['store_nbr',  'family'])['sales'].transform(
            lambda x: x.rolling(7,  min_periods=1).mean()
        )
    return df

train = create_features(train, is_train=True)  # 训练集启用销售特征
test = create_features(test, is_train=False)   # 测试集禁用销售特征

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

# 标准化数值特征
scaler = StandardScaler()