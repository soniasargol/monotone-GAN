# Conditional Sampling With Monotone GANs

Reproducing results in [Conditional Sampling With Monotone GANs](https://arxiv.org/pdf/2006.06755.pdf).

To start running the examples, clone the repository:

```bash
$ cd monotone-gans/
```

## Installation

Run the following commands in the command line to install the necessary libraries and setup the conda envirement:

```bash
conda env create -f environment.yml
source activate mgan
```

### Synthetic examples

Run the script below for creating the synthetic examples. Figures will be saved in the `figs/` directory. Checkpoints of the network weights will be saved in the `checkpoints/` directory.

```bash
$ python src/monotone-gan.py  --example simple --experiment synthetic_4 --equation 4
```

```bash
$ python src/monotone-gan.py  --example simple --experiment synthetic_5 --equation 5
```

```bash
$ python src/monotone-gan.py  --example simple --experiment synthetic_6 --equation 6
```

In all the above cases, the command line input argument `--phase inference` can be used to run the inference phase once there exists a saved checkpoint.

### MNIST Inpainting

Run the command below to train the MNIST inpainting network. The training will take about 12 hours on a small GPU.

```bash
python src/monotone-gan.py  --example mnist --experiment mnist_inpainting --wd 0.0 --batch_size 128 --max_epoch 300 --lr 0.0002
```

Figures will be saved in the `figs/` directory. Checkpoints of the network weights will be saved in the `checkpoints/` directory. The command line input argument `--phase inference` can be used to run the inference phase once there exists a saved checkpoint.


## Author

Sonia Sargolzaei