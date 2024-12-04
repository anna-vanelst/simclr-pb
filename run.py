import torch
import math
from torch import cuda
from torch.utils.data import DataLoader
import torch.optim as optim
from model import CNNet3l, CNNet7l, ProbCNNet3l, ProbCNNet7l
from data import SimCLRAugmentedDataset
from train import trainNNet, trainPNNet
from pb_obj import PBBobj
from risk_certificate import RiskCertificate


class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.data_name = config["data_name"]
        self.batch_size = config["batch_size"]
        self.temperature = config["temperature"]
        self.delta = 0.04
        self.alpha = config["alpha"]
        self.mc_samples = config["mc_samples"]
        self.loader_kargs = (
            {"num_workers": 1, "pin_memory": True} if cuda.is_available() else {})
        self.verbose = True
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.num_classes = 10
        self.prior_dist = "gaussian"
        self.rho_prior = math.log(math.exp(config["sigma_prior"]) - 1.0)
        if self.data_name == "mnist":
            self.prior_model = CNNet3l().to(self.device)
            self.prob_prior_model = ProbCNNet3l(self.rho_prior, prior_dist=self.prior_dist,
                                                device=self.device, init_net=self.prior_model).to(self.device)
        elif self.data_name == "cifar10":
            self.prior_model = CNNet7l().to(self.device)
            self.prob_prior_model = ProbCNNet7l(self.rho_prior, prior_dist=self.prior_dist,
                                                device=self.device, init_net=self.prior_model).to(self.device)
        else:
            print("Invalid data name")
        self.posterior_model = None
        self.risk_cert = RiskCertificate(device=self.device, batch_size=self.batch_size,
                                         temperature=self.temperature, mc_samples=self.mc_samples,
                                         alpha=self.alpha, delta=self.delta)

    def train_prior(self, prior_dataset):
        prior_epochs = self.config["prior_epochs"]
        prior_optimizer = optim.SGD(
            self.prior_model.parameters(),
            lr=self.config["learning_rate_prior"],
            momentum=self.config["momentum_prior"],
        )
        for epoch in range(prior_epochs):
            prior_train = SimCLRAugmentedDataset(
                prior_dataset, name=self.data_name)
            prior_loader = DataLoader(
                prior_train, batch_size=self.batch_size, **self.loader_kargs, shuffle=True
            )
            trainNNet(
                net=self.prior_model,
                optimizer=prior_optimizer,
                epoch=epoch,
                train_loader=prior_loader,
                temperature=self.temperature,
                device=self.device,
                verbose=self.verbose
            )

    def train_prob_prior(self, prior_dataset):
        bound = PBBobj(n_bound=len(prior_dataset), objective="fclassic", kl_penalty=0.000001,
                       device=self.device, batch_size=self.batch_size,
                       temperature=self.temperature, alpha=self.alpha, delta=self.delta)
        prior_optimizer = optim.SGD(
            self.prob_prior_model.parameters(),
            lr=self.config["learning_rate_prior"],
            momentum=self.config["momentum_prior"],
        )
        prior_epochs = self.config["prior_epochs"]
        for epoch in range(prior_epochs):
            prob_prior_train = SimCLRAugmentedDataset(
                prior_dataset, name=self.data_name)
            train_loader = torch.utils.data.DataLoader(
                prob_prior_train, batch_size=self.batch_size, **self.loader_kargs, shuffle=True
            )
            trainPNNet(net=self.prob_prior_model, optimizer=prior_optimizer, pbobj=bound,
                       epoch=epoch, train_loader=train_loader, verbose=self.verbose)

    def init_posterior(self):
        if self.config["prior_type"] == "det":
            init_model = self.prior_model
        elif self.config["prior_type"] == "prob":
            init_model = self.prob_prior_model
        if self.data_name == "mnist":
            self.posterior_model = ProbCNNet3l(self.rho_prior, prior_dist=self.prior_dist,
                                               device=self.device, init_net=init_model).to(self.device)
        elif self.data_name == "cifar10":
            self.posterior_model = ProbCNNet7l(self.rho_prior, prior_dist=self.prior_dist,
                                               device=self.device, init_net=init_model).to(self.device)

    def train_posterior(self, train_dataset):
        self.init_posterior()
        kl_penalty = self.config["kl_penalty"]
        bound = PBBobj(n_bound=len(train_dataset), objective="fclassic", kl_penalty=kl_penalty,
                       device=self.device, batch_size=self.batch_size,
                       temperature=self.temperature, alpha=self.alpha, delta=self.delta)
        posterior_optimizer = optim.SGD(
            self.posterior_model.parameters(),
            lr=self.config["learning_rate_prior"],
            momentum=self.config["momentum_prior"],
        )
        posterior_epochs = self.config["posterior_epochs"]
        for epoch in range(posterior_epochs):
            train = SimCLRAugmentedDataset(train_dataset, name=self.data_name)
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=self.batch_size, **self.loader_kargs, shuffle=True
            )
            trainPNNet(net=self.posterior_model, optimizer=posterior_optimizer, pbobj=bound,
                       epoch=epoch, train_loader=train_loader, verbose=self.verbose)
