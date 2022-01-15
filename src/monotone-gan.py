import argparse
import torch
import numpy as np
from utils import upload_to_dropbox
from train import train_simple, train_mnist
from inference import inference_simple, inference_mnist
np.random.seed(12)
torch.manual_seed(12)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=300,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', dest='lr', type=float, default=2e-4,
                    help='initial learning rate')
parser.add_argument('--wd', dest='wd', type=float,
                    default=1e-4, help='weight decay coefficient')
parser.add_argument('--experiment', dest='experiment',
                    default='experiment', help='experiment name')
parser.add_argument('--example', dest='example',
                    default='simple', help='example')
parser.add_argument('--equation', dest='equation', type=int, default=4,
                    help='4, 5, or 6 for the simple model')
parser.add_argument('--phase', dest='phase',
                    default='train', help='train or inference')
parser.add_argument('--cuda', dest='cuda', type=int, default=1,
                    help='set itto 1 for running on GPU, 0 for CPU')
args = parser.parse_args()

if torch.cuda.is_available() and args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def main():
    """
    Main function that begins training and inference
    """

    if args.phase == 'train':
        if args.example == 'simple':
            train_simple(args, device)
            inference_simple(args, device=device)
        elif args.example == 'mnist':
            train_mnist(args, device)
            inference_mnist(args, device)
        else:
            raise AssertionError()

    elif args.phase == 'inference':
        if args.example == 'simple':
            inference_simple(args, device=device)
        elif args.example == 'mnist':
            inference_mnist(args, device)
        else:
            raise AssertionError()

    upload_to_dropbox(args)

if __name__ == '__main__':
    main()
