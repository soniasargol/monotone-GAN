import torch
import argparse
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from utils import read_checkpoint, create_training_pairs
from architectures import (F_Simple, D_Simple, F_MNIST, D_MNIST)
from forward_operator import Mask, ForwardOperator

sfmt=matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
font = {'family' : 'serif',
        'size'   : 11}
matplotlib.rc('font', **font)
matplotlib.use('Agg')
sns.set(style="whitegrid")


def lf2space(s):
    """
    Convert new lines in a string to spaces.
    """
    return " ".join(s.split("\n"))


def inference_mnist(args, device, num_joint_samples=5, num_cond_samples=1000):

    # Get trainig pairs
    test_loader = create_training_pairs(args)[1]

    # Extract model shape
    for joint, _ in test_loader:
        joint_shape = joint.shape[2:]
        break

    # Noise shape
    y_shape = [14, 14]

    # Forward operator
    mask = Mask(joint_shape, device)

    # Architectures
    F = F_MNIST(np.prod(joint_shape), np.prod(y_shape), n_ch=64).to(device)
    D = D_MNIST(np.prod(joint_shape), n_ch=64).to(device)

    checkpoint_path = os.path.join('checkpoints/', args.experiment,
                                   'checkpoint.pt')

    if os.path.exists(checkpoint_path):
        d_obj_log, g_obj_log = read_checkpoint(F, D, checkpoint_path,
                                               device)

    # Estimated joint samples
    z_y = torch.randn([num_joint_samples, 1,] + y_shape, device=device)
    rand_idx = np.random.choice(joint.shape[0], size=num_joint_samples,
                                replace=False)
    rand_idx = np.sort(rand_idx)
    joint = joint[rand_idx, ...].to(device)
    # Predicted joint samples
    with torch.no_grad():
        joint_x = mask(joint)
        joint_y = F(joint_x, z_y)
    joint_hat = joint_x + torch.nn.functional.pad(joint_y, (3, 3, 3, 3),
                                                  mode='constant',
                                                  value=0.0)
    joint_hat = joint_hat.cpu().numpy()

    # Predicted conditional samples
    cond_samples = torch.zeros([joint.shape[0], num_cond_samples, 1,
                               joint_shape[0], joint_shape[1]])
    true_images = torch.zeros([joint.shape[0], 1,
                               joint_shape[0], joint_shape[1]])
    cropped_images = torch.zeros([joint.shape[0], 1,
                               joint_shape[0], joint_shape[1]])
    with torch.no_grad():
        for j in range(joint.shape[0]):
            z_y = torch.randn([num_cond_samples, 1,] + y_shape,
                              device=device)
            joint_x = mask(joint[j:j+1, ...]).repeat(num_cond_samples, 1, 1, 1)
            joint_y = F(joint_x, z_y)
            cond_samples[j, ...] = (joint_x +
                                    torch.nn.functional.pad(joint_y,
                                                            (3, 3, 3, 3),
                                                            mode='constant',
                                                            value=0.0)
            )
            true_images[j, 0, ...] = joint[j, 0, ...]
            cropped_images[j, 0, ...] = mask(joint[j:j+1, ...])[0, 0, ...]

    cond_samples = cond_samples.cpu().numpy()
    true_images = true_images.cpu().numpy()
    cropped_images = cropped_images.cpu().numpy()

    num_batches = round(60000/args.batch_size)
    itr = np.linspace(0, len(g_obj_log)//num_batches, num=len(g_obj_log))

    # Plotting results
    save_path = os.path.join('figs', args.experiment)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for j in range(joint.shape[0]):
        if not os.path.exists(os.path.join(save_path, str(j))):
            os.makedirs(os.path.join(save_path, str(j)))

    fig = plt.figure("training logs", dpi=150, figsize=(7, 2.5))
    plt.plot(itr, g_obj_log, label="generator", color="#d48955")
    plt.plot(itr, d_obj_log, label="discriminator")
    plt.title("Training objective functions (F)")
    plt.ylabel("Objective")
    plt.xlabel("Epochs")
    plt.grid(False)
    plt.ylim([-0.5, 3])
    plt.legend(loc="lower right", ncol=1, fontsize=8)
    plt.savefig(os.path.join(save_path, "log-F.png"), format="png",
                bbox_inches="tight", dpi=200); plt.close(fig)

    for j in range(num_joint_samples):
        fig = plt.figure("G(z)", dpi=200, figsize=(5, 5))
        plt.imshow(joint_hat[j, 0, ...])
        plt.grid(False)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(os.path.join(save_path, 'sample-K-'
                                + str(j) + '.png'),
                                format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)

    for j in range(joint.shape[0]):
        for i in range(10):
            fig = plt.figure("G(z)", dpi=200, figsize=(5, 5))
            plt.imshow(cond_samples[j, i, 0, ...])
            plt.grid(False)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(os.path.join(os.path.join(save_path, str(j)),
                                     'cond_sample-' + str(i) + '.png'),
                        format='png', bbox_inches='tight', dpi=200)
            plt.close(fig)

        fig = plt.figure("G(z)", figsize=(5, 5))
        plt.imshow(np.mean(cond_samples[j, :, 0, ...], axis=0),
                   vmin=0, vmax=1)
        plt.grid(False)
        plt.title("Conditional mean")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(os.path.join(os.path.join(save_path, str(j)),
                                'cond_mean.png'),
                    format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)

        fig = plt.figure("G(z)", dpi=200, figsize=(5, 5))
        plt.imshow(true_images[j, 0, ...],
                   vmin=0, vmax=1)
        plt.title("Ground truth")
        plt.grid(False)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(os.path.join(os.path.join(save_path, str(j)),
                                    'true_image.png'),
                    format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)

        fig = plt.figure("G(z)", dpi=200, figsize=(5, 5))
        plt.imshow(cropped_images[j, 0, ...],
                   vmin=0, vmax=1)
        plt.grid(False)
        plt.title("Corrupted")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(os.path.join(os.path.join(save_path, str(j)),
                                    'cropped.png'),
                    format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)

        fig = plt.figure("G(z)", figsize=(5, 5))
        plt.imshow(np.std(cond_samples[j, :, 0, ...], axis=0),
                   vmin=0.0, vmax=0.3, cmap="afmhot")
        plt.grid(False)
        plt.title("pSTD")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(os.path.join(os.path.join(save_path, str(j)),
                                'pSTD.png'),
                    format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)


def inference_simple(
    args: argparse.Namespace,
    num_joint_samples: int = 20000,
    num_cond_samples: int = 200,
    device: str = 'cpu'
    ) -> None:
    """
    Creates joint and conditional samples for the simple model.

    Figures will be saved at `os.path.join('figs', args.experiment)`

    Args:
        args: command line arguments.
        device: device to run the model on (cpu or gpu).
    """

    # Get dataset
    x, y = create_training_pairs(args)

    # Setup architectures.
    F = F_Simple(x.shape[1], y.shape[1]).to(device)
    D = D_Simple(x.shape[1], y.shape[1]).to(device)

    # Checkpoint path.
    checkpoint_path = os.path.join('checkpoints', args.experiment,
                                   'checkpoint.pt')

    # Read saved checkpoint and load parameters.
    d_obj_log, g_obj_log = read_checkpoint(F, D, checkpoint_path, device)

    # Initialize latent variables.
    z_y = torch.randn([num_joint_samples, 1], device=device)

    # Sample from marginal x distribution for joint sampling.
    test_idx = np.random.choice(x.shape[0], num_joint_samples, replace=False)
    test_idx.sort()
    joint_x = torch.from_numpy(x[test_idx, :]).to(device)

    # Predicted joint samples
    with torch.no_grad():
        joint_y = F(joint_x, z_y)
    joint_x, joint_y = joint_x.cpu().numpy(), joint_y.cpu().numpy()

    # Pick x samples to condition on for conditional sampling.
    x_test = [torch.Tensor([-1.2]).to(device),
              torch.Tensor([0.]).to(device),
              torch.Tensor([1.2]).to(device)]

    # Condition on x_test and sample from conditional distribution.
    with torch.no_grad():
        posterior_samples = [[] for _ in range(len(x_test))]
        for i, x_ in enumerate(x_test):
            for j in range(num_cond_samples):
                z = torch.randn([1, 1], device=device)
                sample = F(x_.reshape(1, 1), z).cpu().numpy()
                posterior_samples[i].append(sample)
    posterior_samples = np.array(posterior_samples)[:, :, 0, 0].T

    # Creating directories (if does not exist) for saving figures.
    save_path = os.path.join('figs', args.experiment)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Converting components of `x_test` to numpy arrays for plotting.
    x_test = [x_.cpu().numpy() for x_ in x_test]

    # Plotting training losses.
    num_batches = int(x.shape[0]/args.batch_size)
    itr = np.linspace(0, len(g_obj_log)//num_batches, num=len(g_obj_log))

    fig = plt.figure("training logs", figsize=(7, 2.5))
    plt.plot(itr, g_obj_log, label="generator", color="#d48955")
    plt.plot(itr, d_obj_log, label="discriminator")
    plt.title("Training loss functions")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.legend(loc="upper right", ncol=1, fontsize=8)
    plt.savefig(os.path.join(save_path, "log.png"), format="png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)

    # Plotting the predicted joint distribution.
    fig = plt.figure(figsize=(7, 7))
    sns.displot(x=joint_x[:, 0], y=joint_y[:, 0], cmap="Reds", fill=True,
                thresh=0.2, kind="kde")
    plt.ylim([-1.5, 2.5])
    plt.xlim([-4.0, 4.0])
    plt.title("Predicted joint density")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.savefig(os.path.join(save_path, 'predicted_joint_density.png'),
                format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)

    # Plotting the ground truth tagert density.
    fig = plt.figure(figsize=(7, 7))
    sns.displot(x=x[:num_joint_samples, 0], y=y[:num_joint_samples, 0],
                cmap="Blues", fill=True, thresh=0.2, kind="kde")
    plt.ylim([-1.5, 2.5])
    plt.xlim([-4.0, 4.0])
    plt.title("Target joint density")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.savefig(os.path.join(save_path, 'target_joint_density.png'),
                format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)

    # Plotting conditional densities.
    font_prop = matplotlib.font_manager.FontProperties(
        family='serif', style='normal', size=10)
    fig = plt.figure(figsize=(7, 7))
    for idx in range(len(x_test)):
        if args.equation == 6 and idx == 1:
            continue
        ax = sns.kdeplot(posterior_samples[:, idx], fill=True, bw_adjust=0.9,
                         label=r"$x = $ %.1f" % x_test[idx])
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    plt.xlabel("Perturbation", fontproperties=font_prop)
    plt.ylabel("Probability density function", fontproperties=font_prop)
    # plt.xlim([-0.045, 0.045])
    plt.grid(True)
    plt.legend()
    # plt.ylim([0, 125])
    plt.title("Conditional histograms")
    plt.savefig(os.path.join(save_path, 'post_hist.png'),
                        format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)

    with torch.no_grad():
        true_samples = [[] for _ in range(len(x_test))]
        fwd_op = ForwardOperator(equation=args.equation)
        for i, x_ in enumerate(x_test):
            true_samples[i].append(fwd_op(torch.Tensor(x_).repeat(num_cond_samples)).numpy())
    true_samples = np.array(true_samples)[:, 0,  :].T

    # Plotting true conditional densities.
    font_prop = matplotlib.font_manager.FontProperties(
        family='serif', style='normal', size=10)
    fig = plt.figure(figsize=(7, 7))
    for idx in range(len(x_test)):
        if args.equation == 6 and idx == 1:
            continue
        ax = sns.kdeplot(true_samples[:, idx], fill=True, bw_adjust=0.9,
                         label=r"$x = $ %.1f" % x_test[idx])
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    plt.xlabel("Perturbation", fontproperties=font_prop)
    plt.ylabel("Probability density function", fontproperties=font_prop)
    # plt.xlim([-0.045, 0.045])
    plt.grid(True)
    plt.legend()
    # plt.ylim([0, 125])
    plt.title("Conditional histograms")
    plt.savefig(os.path.join(save_path, 'true_post_hist.png'),
                        format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)