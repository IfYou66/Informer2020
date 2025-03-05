from torch.utils.data import DataLoader
import pandas as pd
import warnings
from data.competitionDataloader import Dataset_Favorita
# 忽略所有警告
warnings.filterwarnings("ignore")

# 指定数据目录和 CSV 文件（此处使用预处理后保存的 train.csv）
root_path = './favourite'
data_path = 'train_preprocessed.csv'

# 初始化数据集
dataset = Dataset_Favorita(root_path=root_path, flag='train', size=[30, 15, 15],
                           features='M', data_path=data_path, target='sales',
                           scale=True, inverse=False, timeenc=0, freq='D')

# 输出前五行原始数据
raw_data_path = f"{root_path}/{data_path}"
print("原始数据前五行：")
print(pd.read_csv(raw_data_path).head())

# 如果需要查看数据集处理后的前五个样本
print("\n处理后的数据前五个样本：")
for i in range(5):
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[i]
    print(f"样本 {i+1}:")
    print("Encoder 输入 shape:", seq_x.shape)
    print("Decoder 输出 shape:", seq_y.shape)
