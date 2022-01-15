import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import subprocess
from forward_operator import ForwardOperator

def create_training_pairs(args, n_pairs=50000):
    """
    Generates training pairs
    """
    if args.example == 'simple':
        file_name = 'training_data_' + str(args.equation) + '.hdf5'
        fwd_op = ForwardOperator(equation=args.equation)


        if not os.path.isfile(file_name):
            d_file = h5py.File(file_name, 'w')
            y_samples = d_file.create_dataset("y", [n_pairs, 1],
                                              dtype=np.float32)
            x_samples = d_file.create_dataset("x", [n_pairs, 1],
                                              dtype=np.float32)

            # Add noise to get observed data
            print(' [*] Generates training pairs ...')

            with torch.no_grad():
                for j in range(n_pairs):
                    x = 6.0*torch.rand(1) - 3.0
                    x_samples[j, :] = x.cpu().numpy()[0]
                    y = fwd_op(x)
                    y_samples[j, :] = y.cpu().numpy()[0]

            d_file.close()

        d_file = h5py.File(file_name, 'r')

        return d_file["x"], d_file["y"]

    elif args.example == 'mnist':

        transform = transforms.Compose([
            transforms.CenterCrop(20),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(
            root='data',
            train=True,
            transform=transform,
            download=True
        )
        test_dataset = datasets.MNIST(
            root='data',
            train=False,
            transform=transform
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        return train_loader, test_loader

    else:
        raise AssertionError()

def read_checkpoint(F, D, file_to_load, device):
    """
    Loads network paramteres and training logs.
    """

    assert os.path.isfile(file_to_load)

    if device == torch.device('cpu'):
        checkpoint = torch.load(file_to_load, map_location='cpu')
    else:
        checkpoint = torch.load(file_to_load)

    if 'F_state_dict' in checkpoint:
        F_state_dict = checkpoint['F_state_dict']
        F.load_state_dict(F_state_dict)
        F.eval()

    if 'D_state_dict' in checkpoint:
        D_state_dict = checkpoint['D_state_dict']
        D.load_state_dict(D_state_dict)
        D.eval()

    g_obj_log = checkpoint['g_obj_log']
    d_obj_log = checkpoint['d_obj_log']

    return d_obj_log, g_obj_log


def upload_to_dropbox(args):
    repo_name = "monotone-GANs"
    bash_commands = ["rclone copy -x -v checkpoints/" + args.experiment
                     + " GTDropbox:alisk/" + repo_name + "/checkpoints/"
                     + args.experiment,
                     "rclone copy -x -v figs/" + args.experiment
                     + " GTDropbox:alisk/" + repo_name + "/figs/"
                     + args.experiment,
                     "rclone copy -x -v " + args.experiment + ".out"
                     + " GTDropbox:alisk/" + repo_name + "/checkpoints/"
                     + args.experiment
                    ]
    for commands in bash_commands:
        process = subprocess.Popen(commands.split(), stdout=subprocess.PIPE)
        process.wait()

class CenterCrop(torch.nn.Module):
    """
    Crop the input image
    """
    def __init__(self, in_dim, out_dim):
        super(CenterCrop, self).__init__()

        self.crop_dim = np.zeros([2, 2], dtype=np.int)
        self.crop_dim[0, 0] = (in_dim[0] - out_dim[0])//2
        self.crop_dim[0, 1] = in_dim[0] - out_dim[0] - self.crop_dim[0, 0]
        self.crop_dim[1, 0] = (in_dim[1] - out_dim[1])//2
        self.crop_dim[1, 1] = in_dim[1] - out_dim[1] - self.crop_dim[1, 0]

    def forward(self, x):
        return x[..., self.crop_dim[0, 0]:-self.crop_dim[0, 1],
                 self.crop_dim[1, 0]:-self.crop_dim[1, 1]]