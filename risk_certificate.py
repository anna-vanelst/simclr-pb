import torch
import numpy as np
from tqdm import tqdm
from loss import ZeroOneLoss, SimplifiedContrastiveLoss
from utils import inv_kl


class RiskCertificate:
    """
    A class to compute risk certificates for contrastive learning models using various statistical bounds.

    Attributes:
        device: The device (CPU/GPU) on which computations are performed.
        batch_size: The number of samples per batch.
        temperature: The temperature parameter used in the contrastive loss function.
        mc_samples: The number of Monte Carlo samples to draw during risk estimation.
        alpha: The alpha parameter used in Theorem 1 and 4.
        delta: The confidence level for the PAC-Bayes bound, default is 0.04.
        b_ell_constant: Bound of the contrastive loss.
        epsilon: The epsilon parameter used in Theorem 1.
        b_constant: A constant used in epsilon-modified kl bound calculations.
        c_constant: A constant used in McDiarmid-McAllester bound calculations.
        n_bound: The size of the dataset, used to compute the bounds.

    Methods:
        mcsampling(net, augmented_dataset):
            Performs Monte Carlo sampling to estimate contrastive empirical risk and zero-one 
            contrastive empirical risk.

        forward(net, augmented_dataset, lambda_grid):
            Computes and prints various risk certificates for the given model and dataset.

        iid_catoni_bound(empirical_risk_cont, kl, lambdas):
            Computes the Catoni bound for the given empirical risk and KL divergence.

        iid_kl_bound(empirical_risk_cont, kl):
            Computes the kl bound for the given empirical risk and KL divergence.

        iid_classic_bound(empirical_risk_cont, kl):
            Computes the classic PAC-Bayes bound for the given empirical risk and KL divergence.

        f_divergence_bound(empirical_risk_cont, chi2_divergence):
            Computes the f-divergence bound for the given empirical risk and chi-square divergence.

        epsilon_modified_kl_bound(empirical_risk_cont_epsilon, kl, gamma, alpha):
            Computes the epsilon-modified kl bound (Theorem 1 and 4).

        mcdiarmid_mcallester_bound(empirical_risk_cont, kl):
            Computes the McDiarmid-McAllester bound (Theorem 2 and 5)

        epsilon_modified_exp_bound(empirical_risk_cont_epsilon, kl, lambda_value, gamma, alpha):
            Computes an epsilon-modified Catoni's bound (see appendix).

        _get_list_losses():
            Returns a list of loss functions used for the mc_sampling method.

        _compute_b():
            Computes and returns the b constant used in epsilon-modified kl bounds.

        _compute_c():
            Computes and returns the c constant used in McDiarmid-McAllester bounds.

        _compute_epsilon():
            Computes and returns the epsilon value used in various bounds.

        _compute_gamma():
            Computes and returns the gamma value used in epsilon-modified kl bounds.
    """

    def __init__(
        self,
        device,
        batch_size,
        temperature,
        mc_samples,
        alpha=0.4,
        delta=0.04,
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.delta = delta
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.alpha = alpha
        self.b_ell_constant = 2/self.temperature + np.log(self.batch_size)
        self.epsilon = self._compute_epsilon()
        self.b_constant = self._compute_b()
        self.c_constant = self._compute_c()
        self.n_bound = None

    def mcsampling(self, net, augmented_dataset):
        data_loader = torch.utils.data.DataLoader(
            augmented_dataset, batch_size=self.batch_size, shuffle=False)
        losses = self._get_list_losses()
        res = [0., 0., 0.]
        net.eval()
        for _, list_views in enumerate(tqdm(data_loader)):
            list_views = [data.to(self.device) for data in list_views]
            # cont_loss_eps / cont_loss / zero-one loss
            res_mc = [0.0, 0.0, 0.0]
            for _ in range(self.mc_samples):
                features = [net(data, sample=True) for data in list_views]
                outputs = [loss.forward(features[0], features[1])
                           for loss in losses]
                res_mc = [res_mc[i]+outputs[i] for i in range(3)]
            # we average contrastive loss and contrastive 0-1 error over all MC samples
            res = [res[i] + res_mc[i]/self.mc_samples for i in range(3)]
        # we average cross-entropy and 0-1 error over all batches
        res = [res[i]/self.n_bound for i in range(3)]
        names = ["Contrastive Loss with epsilon",
                 "Contrastive Loss", "Contrastive zero-one risk"]
        for value, name in zip(res, names):
            print(f"{name}: {value}")
        return res

    def forward(self, net, augmented_dataset, lambda_grid=[0.1, 0.5, 1, 5, 10]):
        self.n_bound = len(augmented_dataset)
        with torch.no_grad():
            kl = net.compute_kl()
            kl = kl.cpu().numpy()
            empirical_risk_cont_eps, empirical_risk_cont, empirical_risk_01 = self.mcsampling(
                net, augmented_dataset=augmented_dataset)
        print("GENERAL RES")
        print("n for risk certificates", self.n_bound)
        print("kl div / n", round(float(kl/self.n_bound), 7))
        print("contrastive loss", round(float(empirical_risk_cont), 3))
        print("contrastive loss eps", round(float(empirical_risk_cont_eps), 3))
        print("contrastive zero one risk",  round(float(empirical_risk_01), 3))
        print("b_ell_constant", round(float(self.b_ell_constant), 3))
        print("b_constant", round(float(self.b_constant), 3))
        print("c_constant", round(float(self.c_constant), 3))

        print("CONTRASTIVE LOSS CERTIFICATES")
        print("iid__classic_bound", round(float(self.b_ell_constant *
              self.iid_classic_bound(empirical_risk_cont / self.b_ell_constant, kl)), 3))
        print("iid_kl_bound", round(float(self.b_ell_constant *
              self.iid_kl_bound(empirical_risk_cont / self.b_ell_constant, kl)), 3))
        for lambd in lambda_grid:
            print(f"iid_catoni_bound for lambda {lambd}", round(float(
                self.b_ell_constant * self.iid_catoni_bound(empirical_risk_cont / self.b_ell_constant, kl, lambdas=[lambd])), 3))
        # NOTE: we use kl_div instead of xi_div because kl <= xi
        print("f_divergence_bound", round(float(self.b_ell_constant *
              self.f_divergence_bound(empirical_risk_cont / self.b_ell_constant, kl)), 3))
        print(f"epsilon_modified_kl_bound for alpha {self.alpha}: ", round(float(
            self.b_constant * self.epsilon_modified_kl_bound(empirical_risk_cont_eps / self.b_constant, kl, alpha=self.alpha)), 3))
        print("mcdiarmid_mcallester_bound", round(float(self.c_constant *
              self.mcdiarmid_mcallester_bound(empirical_risk_cont / self.c_constant, kl)), 3))

        print("ZERO-ONE CONTRASTIVE RISK CERTIFICATES")
        gamma = self._compute_gamma()
        print("gamma", round(float(gamma), 3))
        print("iid__classic_bound", round(
            float(self.iid_classic_bound(empirical_risk_01, kl)), 3))
        print("iid_kl_bound", round(
            float(self.iid_kl_bound(empirical_risk_01, kl)), 3))
        print("iid_catoni_bound", round(
            float(self.iid_catoni_bound(empirical_risk_01, kl)), 3))
        for lambd in lambda_grid:
            print(f"iid_catoni_bound for lambda {lambd}", round(
                float(self.iid_catoni_bound(empirical_risk_01, kl, lambdas=[lambd])), 3))
        print("f_divergence_bound", round(
            float(self.f_divergence_bound(empirical_risk_01, kl)), 3))
        print("epsilon_modified_kl_bound", round(
            float(self.epsilon_modified_kl_bound(empirical_risk_01, kl, gamma=gamma)), 3))
        print("mcdiarmid_mcallester_bound", round(
            float(2*self.mcdiarmid_mcallester_bound(empirical_risk_01 / 2, kl)), 3))

    def iid_catoni_bound(self, empirical_risk_cont, kl, lambdas=[0.01, 0.1, 0.5, 1, 5, 10, 100]):
        def catoni_bound(lambda_, empirical_risk_cont, kl, m, n, delta):
            term = -lambda_ * empirical_risk_cont.cpu().numpy() - m * \
                (kl + np.log(1 / delta)) / n
            return (1 - np.exp(term)) / (1 - np.exp(-lambda_))
        m = self.batch_size
        bounds = [
            catoni_bound(
                lambda_, empirical_risk_cont, kl, m, self.n_bound, self.delta
            )
            for lambda_ in lambdas
        ]
        return min(bounds)

    def iid_kl_bound(self, empirical_risk_cont, kl):
        m = self.batch_size
        n = self.n_bound
        delta = self.delta
        complexity_term = (
            kl + np.log(2 * np.sqrt(n) / (delta * np.sqrt(m)))) / n
        return inv_kl(empirical_risk_cont, m * complexity_term)

    def iid_classic_bound(self, empirical_risk_cont, kl):
        m = self.batch_size
        n = self.n_bound
        delta = self.delta
        complexity_term = np.sqrt(
            m * (kl + np.log(2 * np.sqrt(n) / (delta * np.sqrt(m)))) / (2 * n)
        )
        return empirical_risk_cont + complexity_term

    def f_divergence_bound(self, empirical_risk_cont, chi2_divergence):
        m = self.batch_size
        n = self.n_bound
        delta = self.delta
        complexity_term = np.sqrt(
            (m - 1) * (chi2_divergence + 1) / (n * delta))
        return empirical_risk_cont + complexity_term

    def epsilon_modified_kl_bound(self, empirical_risk_cont_epsilon, kl, gamma=0, alpha=0.4):
        n = self.n_bound
        delta = self.delta
        complexity_term = (kl + np.log(np.sqrt(n) / delta)) / n
        left_term = empirical_risk_cont_epsilon + \
            gamma + (delta / 2) ** ((1 - alpha) / alpha)
        term1 = inv_kl(
            left_term,
            complexity_term,
        )
        term2 = (delta / 2) ** (1 / alpha) + gamma
        return term1 + term2

    def mcdiarmid_mcallester_bound(self, empirical_risk_cont, kl):
        n = self.n_bound
        delta = self.delta
        complexity_term = np.sqrt((kl + np.log(2 * n / delta)) / (2 * (n - 1)))
        bound = empirical_risk_cont + complexity_term
        return bound

    def epsilon_modified_exp_bound(self, empirical_risk_cont_epsilon, kl, lambda_value=1, gamma=0, alpha=0.4):
        n = self.n_bound
        delta = self.delta
        complexity_term = (kl + np.log(1 / delta)) / n
        exp_term_numerator = -lambda_value * (empirical_risk_cont_epsilon.cpu(
        ).numpy() + gamma + (delta / 2) ** ((1 - alpha) / alpha)) - complexity_term
        exp_term_denominator = 1 - np.exp(-lambda_value)
        term1 = (1-np.exp(exp_term_numerator)) / exp_term_denominator
        term2 = (delta / 2) ** (1 / alpha) + gamma
        return term1 + term2

    def _get_list_losses(self):
        list_losses = [SimplifiedContrastiveLoss(
            temperature=self.temperature,
            add_epsilon=epsilon,
            delta=self.delta,
            reduction="sum",
            alpha=self.alpha
        ) for epsilon in [True, False]]
        list_losses.append(ZeroOneLoss(reduction="sum"))
        return list_losses

    def _compute_b(self):
        tau = self.temperature
        m = self.batch_size
        b_constant = 1 / tau + np.log(m * np.exp(1 / tau) + self.epsilon)
        return b_constant

    def _compute_c(self):
        m = self.batch_size
        tau = self.temperature
        c_constant = 4/tau + (m-1) * np.log(((m-1) + np.exp(2/tau)) / m)
        return c_constant

    def _compute_epsilon(self):
        m = self.batch_size
        delta = self.delta
        c_bound = np.exp(1 / self.temperature) - np.exp(-1 / self.temperature)
        epsilon = c_bound * \
            np.sqrt(((m-1) * np.log(2 / delta)) / (2 * self.alpha))
        return epsilon

    def _compute_gamma(self):
        m = self.batch_size
        delta = self.delta
        gamma = np.sqrt((np.log(2 / delta)) / (2 * (m-1) * self.alpha))
        return gamma
