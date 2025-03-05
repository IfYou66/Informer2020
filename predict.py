import argparse
import os
import torch
import numpy as np

from exp.exp_informer import Exp_Informer


def main():
    parser = argparse.ArgumentParser(description='Predict using Informer on Favourite dataset')

    # 基本参数配置（请根据实际情况调整默认值）
    parser.add_argument('--model', type=str, default='informer')
    parser.add_argument('--data', type=str, default='Favourite')
    parser.add_argument('--root_path', type=str, default='./data/favourite/')
    parser.add_argument('--data_path', type=str, default='train_preprocessed.csv')
    parser.add_argument('--features', type=str, default='MS',
                        help='M: multivariate predict multivariate; S: univariate predict univariate; MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='sales')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)

    parser.add_argument('--enc_in', type=int, default=18)
    parser.add_argument('--dec_in', type=int, default=18)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--s_layers', type=str, default='3,2,1')
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--padding', type=int, default=0)
    # distil 默认 True，这里通过 --distil 指定 False
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', default=False)
    # 这里设置 do_predict 为 True，表示我们需要预测
    parser.add_argument('--do_predict', action='store_true', default=True)
    parser.add_argument('--mix', action='store_false', default=True)
    parser.add_argument('--cols', type=str, nargs='+', default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='predict')
    parser.add_argument('--loss', type=str, default='rmsle')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--inverse', action='store_true', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    args = parser.parse_args()

    # 设置 GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 处理 s_layers 和 freq 参数
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    # 数据集参数设置
    data_parser = {
        'Favourite': {'data': 'train_preprocessed.csv', 'T': 'sales', 'M': [17, 17, 1], 'S': [1, 1, 1],
                      'MS': [17, 17, 1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    print('Args:', args)

    # 初始化实验对象
    exp = Exp_Informer(args)

    # 构造 setting 字符串（用于文件命名等）
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}'.format(
        args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn,
        args.factor, args.embed, args.distil, args.mix, args.des)

    # 调用 predict 方法，并设置 load=True 来载入训练好的模型参数
    exp.predict(setting, load=True)


if __name__ == '__main__':
    main()
