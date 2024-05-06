import argparse

import time

import torch

from exp.exp_main import Exp_MAIN
from utils.tools import setSeed

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='IDEA',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/illness/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='national_illness.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./result', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')

    # optimizationF
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--seeds', type=int, default=[2022, 2023, 2024], nargs='+',
                        help='number of hidden layers in projector')

    # IDEA
    parser.add_argument('--zc_dim', type=int, default=7, help='num of encoder layers')
    parser.add_argument('--zd_dim', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--zd_kl_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--zc_kl_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--hmm_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--rec_weight', type=float, default=0.5, help='latent dimension of koopman embedding')
    parser.add_argument('--n_class', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--lags', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--embedding_dim', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--is_bn', action='store_true', default=False, help='num of encoder layers')
    parser.add_argument('--dynamic_dim', type=int, default=128, help='latent dimension of koopman embedding')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of en/decoder')
    parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers of en/decoder')
    parser.add_argument('--pre_epoches', type=int, default=1, help='train epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_MAIN

    for seed in args.seeds:
        setSeed(seed)
        exp = Exp(args)  # set experiments
        exp.seed = seed
        exp.train()
        exp.test()
        torch.cuda.empty_cache()
