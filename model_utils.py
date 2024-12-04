# Credit: M. Perez-Ortiz - https://github.com/mperezortiz/PBB
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1.0 - eps), max=(1.0 - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device="cuda", fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class Laplace(nn.Module):
    """Implementation of a Laplace random variable, using softplus for
    the scale parameter and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Laplace distr.

    rho : Tensor of floats
        Scale parameter for the distribution (to be transformed
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the distribution is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device="cuda", fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999 * torch.rand(self.scale.size()) -
                   0.49999).to(self.device)
        result = self.mu - torch.mul(
            torch.mul(self.scale, torch.sign(epsilon)), torch.log(
                1 - 2 * torch.abs(epsilon))
        )
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces distr. (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div


class Linear(nn.Module):
    """Implementation of a Linear layer (reimplemented to use
    truncated normal as initialisation for fair comparison purposes)

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, in_features, out_features, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1 / np.sqrt(in_features)

        # same initialisation as before for the prob layer
        self.weight = nn.Parameter(
            trunc_normal_(
                torch.Tensor(out_features, in_features),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            ),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)


class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(
        self,
        in_features,
        out_features,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        init_prior="random",
        init_layer=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1 / np.sqrt(in_features)

        # INITIALISE PRIOR
        # this means prior is uninformed
        if not init_layer:
            # initalise prior to zeros and rho_prior
            if init_prior == "zeros":
                bias_mu_prior = torch.zeros(out_features)
                weights_mu_prior = torch.zeros(out_features, in_features)
                weights_rho_prior = torch.ones(
                    out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior
            # initialise prior to random weights and rho_prior
            elif init_prior == "random":
                weights_mu_prior = trunc_normal_(
                    torch.Tensor(out_features, in_features),
                    0,
                    sigma_weights,
                    -2 * sigma_weights,
                    2 * sigma_weights,
                )
                bias_mu_prior = torch.zeros(out_features)
                weights_rho_prior = torch.ones(
                    out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior
            else:
                raise RuntimeError(f"Wrong type of prior initialisation!")
        # informed prior
        else:
            # if init layer is probabilistic
            if hasattr(init_layer.weight, "rho"):
                weights_mu_prior = init_layer.weight.mu
                bias_mu_prior = init_layer.bias.mu
                weights_rho_prior = init_layer.weight.rho
                bias_rho_prior = init_layer.bias.rho
            # if init layer for prior is not probabilistic
            else:
                weights_mu_prior = init_layer.weight
                bias_mu_prior = init_layer.bias
                weights_rho_prior = torch.ones(
                    out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior

        # INITIALISE POSTERIOR
        # WE ASSUME THAT ALWAYS POSTERIOR WILL BE INITIALISED TO PRIOR (UNLESS PRIOR IS INITIALISED TO ALL ZEROS)
        if init_prior == "zeros":
            weights_mu_init = trunc_normal_(
                torch.Tensor(out_features, in_features),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            )
            bias_mu_init = torch.zeros(out_features)
            weights_rho_init = torch.ones(
                out_features, in_features) * rho_prior
            bias_rho_init = torch.ones(out_features) * rho_prior
        # initialise to prior
        else:
            weights_mu_init = weights_mu_prior
            bias_mu_init = bias_mu_prior
            weights_rho_init = weights_rho_prior
            bias_rho_init = bias_rho_prior

        if prior_dist == "gaussian":
            dist = Gaussian
        elif prior_dist == "laplace":
            dist = Laplace
        else:
            raise RuntimeError(f"Wrong prior_dist {prior_dist}")

        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(
            weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=False
        )
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_prior.clone(), device=device, fixed=True
        )
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_prior.clone(), device=device, fixed=True
        )

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(
                self.bias_prior
            )

        return F.linear(input, weight, bias)

    def __deepcopy__(self, memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        self.kl_div = self.kl_div.detach().clone()
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method
        return cp


class ProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    init_layer : Linear object
        Linear layer object used to initialise the prior

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        stride=1,
        padding=0,
        dilation=1,
        init_prior="random",
        init_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.init_prior = init_prior

        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1 / np.sqrt(in_features)

        # INITIALISE PRIOR
        # this means prior is uninformed
        if not init_layer:
            # initalise prior to zeros and rho_prior
            if init_prior == "zeros":
                bias_mu_prior = torch.zeros(out_channels)
                weights_mu_prior = torch.zeros(
                    out_channels, in_channels, *self.kernel_size)
                weights_rho_prior = (
                    torch.ones(out_channels, in_channels, *
                               self.kernel_size) * rho_prior
                )
                bias_rho_prior = torch.ones(out_channels) * rho_prior
            # initialise prior to random weights and rho_prior
            elif init_prior == "random":
                weights_mu_prior = trunc_normal_(
                    torch.Tensor(out_channels, in_channels, *self.kernel_size),
                    0,
                    sigma_weights,
                    -2 * sigma_weights,
                    2 * sigma_weights,
                )
                bias_mu_prior = torch.zeros(out_channels)
                weights_rho_prior = (
                    torch.ones(out_channels, in_channels, *
                               self.kernel_size) * rho_prior
                )
                bias_rho_prior = torch.ones(out_channels) * rho_prior
            else:
                raise RuntimeError(f"Wrong type of prior initialisation!")
        # informed prior
        else:
            # if init layer is probabilistic
            if hasattr(init_layer.weight, "rho"):
                weights_mu_prior = init_layer.weight.mu
                bias_mu_prior = init_layer.bias.mu
                weights_rho_prior = init_layer.weight.rho
                bias_rho_prior = init_layer.bias.rho
            # if init layer for prior is not probabilistic
            else:
                weights_mu_prior = init_layer.weight
                bias_mu_prior = init_layer.bias
                weights_rho_prior = (
                    torch.ones(out_channels, in_channels, *
                               self.kernel_size) * rho_prior
                )
                bias_rho_prior = torch.ones(out_channels) * rho_prior

        # INITIALISE POSTERIOR
        # WE ASSUME THAT ALWAYS PRIOR WILL BE INITIALISED TO PRIOR (UNLESS PRIOR IS INITIALISED TO ALL ZEROS)
        if init_prior == "zeros":
            weights_mu_init = trunc_normal_(
                torch.Tensor(out_channels, in_channels, *self.kernel_size),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            )
            bias_mu_init = torch.zeros(out_channels)
            weights_rho_init = torch.ones(
                out_channels, in_channels, *self.kernel_size) * rho_prior
            bias_rho_init = torch.ones(out_channels) * rho_prior
        # initialise to prior
        else:
            weights_mu_init = weights_mu_prior
            bias_mu_init = bias_mu_prior
            weights_rho_init = weights_rho_prior
            bias_rho_init = bias_rho_prior

        if prior_dist == "gaussian":
            dist = Gaussian
        elif prior_dist == "laplace":
            dist = Laplace
        else:
            raise RuntimeError(f"Wrong prior_dist {prior_dist}")

        self.weight = dist(
            weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=False
        )
        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_prior.clone(), device=device, fixed=True
        )
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_prior.clone(), device=device, fixed=True
        )

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(
                self.bias_prior
            )

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def __deepcopy__(self, memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        self.kl_div = self.kl_div.detach().clone()
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method
        return cp
