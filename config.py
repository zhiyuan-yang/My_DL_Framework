import torch
import torch.nn as nn
import argparse
import datetime


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
        
parser = argparse.ArgumentParser(description='Training Option')        
 ### log setting
parser.add_argument('--log_file_name', type=str, default='./log/' + datetime.datetime.now().strftime('%Y-%m-%d') + '.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='myLogger',
                    help='Logger name')

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='The number of GPU used in training')

### dataloader setting
parser.add_argument('--num_workers', type=int, default=4,
                    help='The number of workers when loading data')

### model setting


### loss setting


### optimizer setting
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr_rate_dis', type=float, default=1e-4,
                    help='Learning rate of discriminator')
parser.add_argument('--lr_rate_lte', type=float, default=1e-5,
                    help='Learning rate of LTE')
parser.add_argument('--decay', type=float, default=50,
                    help='Learning rate decay type')
parser.add_argument('--gamma', type=float, default=1/3,
                    help='Learning rate decay factor for step decay')

### training setting
parser.add_argument('--batch_size', type=int, default=12,
                    help='Training batch size')
parser.add_argument('--train_crop_size', type=int, default=40,
                    help='Training data crop size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='The number of training epochs')


### validation setting

parser.add_argument('--val_freq', type=int, default=5,
                    help='Validation Frequency')


args = parser.parse_args()       
