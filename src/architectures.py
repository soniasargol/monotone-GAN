import torch

class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        """
        A utility for reshaping in torch.nn.Sequential
        """
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

class F_MNIST(torch.nn.Module):
    def __init__(self, dim_d, dim_m, n_ch=64):
        super(F_MNIST, self).__init__()
        """
        The network F in https://arxiv.org/pdf/2006.06755.pdf
        """
        self.dim_m = dim_m
        self.dim_d = dim_d

        self.model = torch.nn.Sequential(
                        Reshape(-1, dim_m + dim_d),
                        torch.nn.Linear(dim_m + dim_d, 1000, bias=True),
                        torch.nn.LeakyReLU(),

                        Reshape(-1, 1000, 1, 1),

                        torch.nn.ConvTranspose2d(1000, n_ch * 4,
                                                 kernel_size=4, stride=1,
                                                 padding=0, bias=False),
                        torch.nn.BatchNorm2d(n_ch * 4),
                        torch.nn.ReLU(True),

                        torch.nn.ConvTranspose2d(n_ch * 4, n_ch * 2,
                                                 kernel_size=4, stride=2,
                                                 padding=1, bias=False),
                        torch.nn.BatchNorm2d(n_ch * 2),
                        torch.nn.ReLU(True),

                        torch.nn.ConvTranspose2d(n_ch * 2, n_ch,
                                                 kernel_size=4, stride=1,
                                                 padding=2, bias=False),
                        torch.nn.BatchNorm2d(n_ch),
                        torch.nn.ReLU(True),

                        torch.nn.ConvTranspose2d(n_ch, 1,
                                                 kernel_size=4, stride=2,
                                                 padding=1, bias=False),
                        torch.nn.Sigmoid()
                        )

    def forward(self, x, z_y):
        return self.model(torch.cat((x.reshape(-1, self.dim_d),
                                     z_y.reshape(-1, self.dim_m)), dim=-1))

class D_MNIST(torch.nn.Module):
    def __init__(self, dim_joint, n_ch=4):
        super(D_MNIST, self).__init__()
        """
        The discriminator network (f) in https://arxiv.org/pdf/2006.06755.pdf
        """
        self.model = torch.nn.Sequential(
                        torch.nn.Conv2d(1, n_ch, 4, 2, 1, bias=False),
                        torch.nn.LeakyReLU(0.2, inplace=True),

                        torch.nn.Conv2d(n_ch, n_ch*2, 4, 2, 2, bias=False),
                        torch.nn.BatchNorm2d(n_ch*2),
                        torch.nn.LeakyReLU(0.2, inplace=True),

                        torch.nn.Conv2d(n_ch*2, n_ch*4, 4, 2, 2,bias=False),
                        torch.nn.BatchNorm2d(n_ch*4),
                        torch.nn.LeakyReLU(0.2, inplace=True),

                        torch.nn.Conv2d(n_ch*4, n_ch*8, 4, 2, 1, bias=False),
                        torch.nn.BatchNorm2d(n_ch*8),
                        torch.nn.LeakyReLU(0.2, inplace=True),

                        torch.nn.Conv2d(n_ch*8, 1, 2, 1, 0, bias=False),
                        # torch.nn.Sigmoid(),
                        Reshape(-1),
                        )

    def forward(self, joint):
        return self.model(joint)


class K_Simple(torch.nn.Module):
    def __init__(self, dim_d):
        super(K_Simple, self).__init__()
        """
        The network K in https://arxiv.org/pdf/2006.06755.pdf
        """
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(dim_d, 100, bias=False),
                        torch.nn.BatchNorm1d(100),
                        torch.nn.LeakyReLU(),

                        torch.nn.Linear(100, 200, bias=False),
                        torch.nn.BatchNorm1d(200),
                        torch.nn.LeakyReLU(),

                        torch.nn.Linear(200, 100, bias=False),
                        torch.nn.BatchNorm1d(100),
                        torch.nn.LeakyReLU(),

                        torch.nn.Linear(100, dim_d)
                        )

    def forward(self, z_y):
        return self.model(z_y)

class F_Simple(torch.nn.Module):
    def __init__(self, dim_d, dim_m):
        super(F_Simple, self).__init__()
        """
        The network F in https://arxiv.org/pdf/2006.06755.pdf
        """
        self.model = torch.nn.Sequential(
                        Reshape(-1, dim_m + dim_d),

                        torch.nn.Linear(dim_m + dim_d, 256, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.2),

                        torch.nn.Linear(256, 512, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.2),

                        torch.nn.Linear(512, 128, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.2),

                        torch.nn.Linear(128, dim_m),
                        )

    def forward(self, x, z_y):
        return self.model(torch.cat((x, z_y), dim=-1))

class D_Simple(torch.nn.Module):
    def __init__(self, dim_d, dim_m):
        super(D_Simple, self).__init__()
        """
        The discriminator network (f) in https://arxiv.org/pdf/2006.06755.pdf
        """
        self.model = torch.nn.Sequential(
                        Reshape(-1, dim_m + dim_d),

                        torch.nn.Linear(dim_m + dim_d, 256, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.2),

                        torch.nn.Linear(256, 512, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.2),

                        torch.nn.Linear(512, 128, bias=True),
                        torch.nn.LeakyReLU(negative_slope=0.2),

                        torch.nn.Linear(128, 1, bias=True),
                        )

    def forward(self, x, y):
        return self.model(torch.cat((x, y), dim=-1))
