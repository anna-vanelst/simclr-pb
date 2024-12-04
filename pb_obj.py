from loss import SimplifiedContrastiveLoss, ZeroOneLoss
import torch
import numpy as np


class PBBobj:
    def __init__(
        self,
        n_bound,
        objective,
        kl_penalty,
        device,
        batch_size,
        temperature,
        alpha=0.4,
        delta=0.04,
    ):
        super().__init__()
        self.objective = objective
        self.device = device
        self.kl_penalty = kl_penalty
        self.n_bound = n_bound
        self.temperature = temperature
        self.delta = delta
        self.batch_size = batch_size
        self.alpha = alpha
        self.epsilon = self._compute_epsilon()
        self.b_constant = self._compute_b()

    def train_obj(self, net, list_views):
        kl = net.compute_kl()
        contrastive_loss = SimplifiedContrastiveLoss(
            temperature=self.temperature,
            add_epsilon=False,
            delta=self.delta,
            reduction="mean",
        )
        view0, view1 = list_views
        empirical_risk = contrastive_loss.forward(view0, view1)/self.b_constant
        train_obj = self.bound(empirical_risk, kl)
        zero_one_loss = ZeroOneLoss()
        loss_0_1 = zero_one_loss.forward(view0, view1)
        return train_obj, kl/self.n_bound, empirical_risk, loss_0_1

    def bound(self, empirical_risk, kl):
        train_size = self.n_bound
        if self.objective == "fclassic":
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(
                kl + np.log((np.sqrt(train_size)) / self.delta), 2 * train_size
            )
            train_obj = empirical_risk + torch.sqrt(kl_ratio)
        elif self.objective == "bbb":
            train_obj = empirical_risk + self.kl_penalty * (kl / train_size)
        else:
            raise RuntimeError(f"Wrong objective {self.objective}")
        return train_obj

    def _compute_b(self):
        tau = self.temperature
        m = self.batch_size
        b_constant = 1 / tau + np.log(m * np.exp(1 / tau) + self.epsilon)
        return b_constant

    def _compute_epsilon(self):
        m = self.batch_size
        delta = self.delta
        c_bound = np.exp(1 / self.temperature) - np.exp(-1 / self.temperature)
        epsilon = c_bound * \
            np.sqrt(((m-1) * np.log(2 / delta)) / (2 * self.alpha))
        return epsilon
