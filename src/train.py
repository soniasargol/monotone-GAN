import torch
import numpy as np
import time
import os
from utils import create_training_pairs, read_checkpoint
from architectures import F_Simple, D_Simple, F_MNIST, D_MNIST
from forward_operator import Mask

def train_mnist(args, device):

    # Get training pairs
    train_loader = create_training_pairs(args)[0]

    # Extract model shape
    for joint, _ in train_loader:
        joint_shape = joint.shape[2:]
        break

    # Noise shape
    y_shape = [14, 14]

    # Forward operator
    mask = Mask(joint_shape, device)

    # Architectures
    F = F_MNIST(np.prod(joint_shape), np.prod(y_shape), n_ch=64).to(device)
    D = D_MNIST(np.prod(joint_shape), n_ch=64).to(device)

    max_epoch = args.max_epoch
    start_time = time.time()
    f_obj_log = []
    d_obj_log = []
    batch_size = args.batch_size
    num_batches = round(60000/batch_size)

    # Optimization algorithms
    G_params = list(F.parameters())
    optim_F = torch.optim.Adam(G_params, lr=args.lr,
                               betas=(0.5, 0.999),
                               weight_decay=args.wd)
    optim_D = torch.optim.Adam(D.parameters(), lr=args.lr,
                               betas=(0.5, 0.999),
                               weight_decay=args.wd)

    #Learning rate scheduler
    scheduler_F = torch.optim.lr_scheduler.MultiStepLR(optim_F,
                            milestones=[100, 300, 480], gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optim_D,
                            milestones=[100, 300, 480], gamma=0.1)

    # Directtory to checkpints
    checkpoint_path = os.path.join('checkpoints/', args.experiment)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Train network F
    epoch = 0
    while epoch < max_epoch:

        for itr, (joint, _) in enumerate(train_loader):

            # Load training data into memory
            joint = joint.to(device)

            # Apply mask
            x = mask(joint)
            y = joint[..., 3:-3, 3:-3]

            z_y = torch.randn_like(y)

            # Predicted joint samples
            y_hat = F(x, z_y)

            # Average monotonicity loss
            z_y_ = torch.randn_like(y)
            y_hat_ = F(x, z_y_)
            monoton_obj = 0.01 * torch.dot((y_hat - y_hat_).reshape(-1),
                                           (z_y - z_y_).reshape(-1))
            joint_hat = x + torch.nn.functional.pad(y_hat, (3, 3, 3, 3),
                                                    mode='constant',
                                                    value=0.0)

            g_obj = 0.5 * torch.norm(D(joint_hat) - 1.0)**2 - monoton_obj
            g_obj /= batch_size

            # Compute the gradient w.r.t generator parameters
            grad = torch.autograd.grad(g_obj, G_params,
                                       retain_graph=True)
            for param, grad in zip(G_params, grad):
                param.grad = grad

            # Update network parameters
            optim_F.step()

            # Update D
            d_obj = 0.5 * (torch.norm(D(joint) - 1.0)**2 +
                           torch.norm(D(joint_hat))**2)
            d_obj /= batch_size

            # Compute the gradient w.r.t discriminator parameters
            grad = torch.autograd.grad(d_obj, D.parameters(),
                                       retain_graph=False)
            for param, grad in zip(D.parameters(), grad):
                param.grad = grad

            # Update network parameters
            optim_D.step()

            f_obj_log.append(g_obj.detach().item())
            d_obj_log.append(d_obj.detach().item())

            if itr%10 == 0:
                print(("Epoch: [%d/%d] | iteration: [%d/%d] | time: %4.4f |"
                       "G obj: %4.4f | D obj: %4.4f" %
                      (epoch+1, max_epoch, itr+1, num_batches, time.time()
                       - start_time, f_obj_log[-1], d_obj_log[-1])))

        scheduler_F.step()
        scheduler_D.step()

        if (epoch%10 == 0) or (epoch == max_epoch-1):
            torch.save({'F_state_dict': F.state_dict(),
                        'D_state_dict': D.state_dict(),
                        'g_obj_log': f_obj_log,
                        'd_obj_log': d_obj_log},
                        os.path.join(checkpoint_path,
                                     'checkpoint.pt'))
            torch.save({'F_state_dict': F.state_dict(),
                        'D_state_dict': D.state_dict(),
                        'g_obj_log': f_obj_log,
                        'd_obj_log': d_obj_log},
                        os.path.join(checkpoint_path,
                                     'checkpoint_epoch-' + str(epoch) + '.pt'))
        epoch += 1


def train_simple(args, device):

    # Get trainig pairs
    x, y = create_training_pairs(args, n_pairs=50000)

    # Architectures
    F = F_Simple(x.shape[1], y.shape[1]).to(device)
    D = D_Simple(x.shape[1], y.shape[1]).to(device)

    # Training loop
    max_epoch = args.max_epoch
    start_time = time.time()
    d_obj_log = []
    g_obj_log = []
    batch_size = args.batch_size

    # Optimization algorithms
    G_params = list(F.parameters())
    optim_G = torch.optim.Adam(G_params, lr=args.lr, weight_decay=args.wd)
    optim_D = torch.optim.Adam(D.parameters(), lr=args.lr, weight_decay=args.wd)

    # Extracting training indices
    train_idx = np.random.choice(x.shape[0], x.shape[0], replace=False)
    num_batches = int(x.shape[0]/batch_size)

    checkpoint_path = os.path.join('checkpoints/', args.experiment)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    epoch = 0
    x = torch.from_numpy(x[...]).to(device)
    y = torch.from_numpy(y[...]).to(device)
    while epoch < max_epoch:

        # Shuffle training set every epoch
        np.random.shuffle(train_idx)

        for itr in range(num_batches):

            # Randomly select training indices
            idx = train_idx[itr*batch_size:(itr+1)*batch_size]
            idx.sort()

            # Load training pairs into memory
            x_train = x[idx, :]
            y_train = y[idx, :]

            z_y = torch.randn_like(y_train)

            # Predicted joint samples
            y_hat = F(x_train, z_y)

            # Average monotonicity loss
            z_y_ = torch.randn_like(y_train)
            y_hat_ = F(x_train, z_y_)
            monoton_obj = 0.01 * torch.dot((y_hat - y_hat_).reshape(-1),
                                           (z_y - z_y_).reshape(-1))

            g_obj = 0.5 * torch.norm(D(x_train, y_hat) - 1.0)**2 - monoton_obj
            g_obj /= batch_size

            # Compute the gradient w.r.t generator parameters
            grad = torch.autograd.grad(g_obj, G_params,
                                       retain_graph=True)
            for param, grad in zip(G_params, grad):
                param.grad = grad

            # Update network parameters
            optim_G.step()

            # Update D
            d_obj = 0.5 * (torch.norm(D(x_train, y_train) - 1.0)**2 +
                           torch.norm(D(x_train, y_hat))**2)
            d_obj /= batch_size

            # Compute the gradient w.r.t discriminator parameters
            grad = torch.autograd.grad(d_obj, D.parameters(),
                                       retain_graph=False)
            for param, grad in zip(D.parameters(), grad):
                param.grad = grad

            # Update network parameters
            optim_D.step()

            g_obj_log.append(g_obj.detach().item())
            d_obj_log.append(d_obj.detach().item())

            print(("Epoch: [%d/%d] | iteration: [%d/%d] | time: %4.4f |"
                    "G obj: %4.4f | D obj: %4.4f" %
                    (epoch+1, max_epoch, itr+1, num_batches, time.time()
                    - start_time, g_obj_log[-1], d_obj_log[-1])))

        if (epoch%10 == 0) or (epoch == max_epoch - 1):
            torch.save({'F_state_dict': F.state_dict(),
                        'D_state_dict': D.state_dict(),
                        'g_obj_log': g_obj_log,
                        'd_obj_log': d_obj_log},
                        os.path.join(checkpoint_path,
                                     'checkpoint.pt'))

        epoch += 1
