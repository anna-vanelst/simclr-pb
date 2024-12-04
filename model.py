# Credit: M. Perez-Ortiz - https://github.com/mperezortiz/PBB
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import ProbLinear, ProbConv2d


class CNNet7l(nn.Module):
    """
    Convolutional neural network (CNN) designed for experiments on the CIFAR-10 dataset. 
    It includes seven layers of convolutional operations, followed by a two-layer projection head.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(4096, 2048)
        self.fcl2 = nn.Linear(2048, 2048)
        self.fcl3 = nn.Linear(2048, 128)

    def forward(self, x, projection=True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcl1(x))
        if projection:
            x = F.relu(self.fcl2(x))
            x = self.fcl3(x)
        return x


class ProbCNNet7l(nn.Module):
    """
    Probabilistic version of CNNet7l, designed for experiments on the CIFAR-10 dataset.
    """

    def __init__(self, rho_prior, prior_dist, device="cuda", init_net=None, init_prior="weights"):
        super().__init__()
        self.conv1 = ProbConv2d(
            in_channels=3,
            out_channels=32,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            kernel_size=3,
            padding=1,
            init_layer=init_net.conv1 if init_net else None,
            init_prior=init_prior,
        )
        self.conv2 = ProbConv2d(
            in_channels=32,
            out_channels=64,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            kernel_size=3,
            padding=1,
            init_layer=init_net.conv2 if init_net else None,
            init_prior=init_prior,
        )
        self.conv3 = ProbConv2d(
            in_channels=64,
            out_channels=128,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            kernel_size=3,
            padding=1,
            init_layer=init_net.conv3 if init_net else None,
            init_prior=init_prior,
        )
        self.conv4 = ProbConv2d(
            in_channels=128,
            out_channels=128,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            kernel_size=3,
            padding=1,
            init_layer=init_net.conv4 if init_net else None,
            init_prior=init_prior,
        )
        self.conv5 = ProbConv2d(
            in_channels=128,
            out_channels=256,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            kernel_size=3,
            padding=1,
            init_layer=init_net.conv5 if init_net else None,
            init_prior=init_prior,
        )
        self.conv6 = ProbConv2d(
            in_channels=256,
            out_channels=256,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            kernel_size=3,
            padding=1,
            init_layer=init_net.conv6 if init_net else None,
            init_prior=init_prior,
        )

        self.fcl1 = ProbLinear(
            4096,
            2048,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fcl1 if init_net else None,
            init_prior=init_prior,
        )
        self.fcl2 = ProbLinear(
            2048,
            2048,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fcl2 if init_net else None,
            init_prior=init_prior,
        )
        self.fcl3 = ProbLinear(
            2048,
            128,
            rho_prior=rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fcl3 if init_net else None,
            init_prior=init_prior,
        )

    def forward(self, x, sample=False, projection=True):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcl1(x, sample))
        if projection:
            x = F.relu(self.fcl2(x, sample))
            x = self.fcl3(x, sample)
        return x

    def compute_kl(self):
        return (
            self.conv1.kl_div
            + self.conv2.kl_div
            + self.conv3.kl_div
            + self.conv4.kl_div
            + self.conv5.kl_div
            + self.conv6.kl_div
            + self.fcl1.kl_div
            + self.fcl2.kl_div
            + self.fcl3.kl_div
        )


class CNNet3l(nn.Module):
    """
    Convolutional neural network (CNN) designed for experiments on the MNIST dataset. 
    It includes three layers of convolutional operations, followed by a two-layer projection head.
    """

    def __init__(self, neurons=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 128)

    def forward(self, x, projection=True):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if projection:
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
        return x


class ProbCNNet3l(nn.Module):
    """
    Probabilistic version of CNNet3l, designed for experiments on the MNIST dataset.
    """

    def __init__(
        self, rho_prior, prior_dist="gaussian", device="cuda", init_net=None, init_prior="weights", neurons=512
    ):
        super().__init__()

        self.conv1 = ProbConv2d(
            1,
            32,
            3,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.conv1 if init_net else None,
            init_prior=init_prior,
        )
        self.conv2 = ProbConv2d(
            32,
            64,
            3,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.conv2 if init_net else None,
            init_prior=init_prior,
        )
        self.fc1 = ProbLinear(
            9216,
            neurons,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fc1 if init_net else None,
            init_prior=init_prior,
        )
        self.fc2 = ProbLinear(
            neurons,
            neurons,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fc2 if init_net else None,
            init_prior=init_prior,
        )
        self.fc3 = ProbLinear(
            neurons,
            128,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fc3 if init_net else None,
            init_prior=init_prior,
        )

    def forward(self, x, sample=False, projection=True):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        if projection:
            x = self.fc2(x, sample)
            x = F.relu(x)
            x = self.fc3(x, sample)
        return x

    def compute_kl(self):
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div + self.fc3.kl_div
