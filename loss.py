import numpy as np
import torch


class ContrastiveLoss:
    """
    Calculates the contrastive loss.

    Args:
        temperature (float): Scaling factor for the similarity scores. Default is 0.1.
        add_epsilon (bool): Whether to compute the epsilon modified loss. Default is False.
        delta (float): A parameter used in epsilon computation (see Theorem 1). Default is 0.02.
        reduction (str): Options are 'mean' or 'sum'. Default is 'mean'.
        alpha (float): A parameter used in epsilon computation (see Theorem 1). Default is 0.1.

    Methods:
        forward(first_view, second_view):
            Calculates the final symmetric contrastive loss.

        contrastive_loss_single_view(view0, view1):
            Calculates the contrastive loss for a given order of parameters.

        _compute_epsilon(m):
            Computes epsilon for the epsilon-modified contrastive loss.
    """

    def __init__(
        self, temperature: float = 0.1, add_epsilon=False, delta=0.04, reduction="mean", alpha=0.1
    ):
        self.temperature = temperature
        self.add_epsilon = add_epsilon
        self.delta = delta
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, first_view: torch.Tensor, second_view: torch.Tensor) -> torch.FloatTensor:
        """
        Calculates the final contrastive loss for both views.

        Args:
            first_view (torch.Tensor): The first augmented view. Shape: (batch_size, embedding_dim).
            second_view (torch.Tensor): The second augmented view. Shape: (batch_size, embedding_dim).

        Returns:
            torch.FloatTensor: The contrastive loss.
        """
        first_loss = self.contrastive_loss_single_view(first_view, second_view)
        second_loss = self.contrastive_loss_single_view(
            second_view, first_view)
        losses = [first_loss, second_loss]
        final_loss = torch.stack(losses).mean()
        return final_loss

    def contrastive_loss_single_view(
        self, view0: torch.Tensor, view1: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Calculates the contrastive loss for a specific order of the views.

        Args:
            view0 (torch.Tensor): The first view. Shape: (batch_size, embedding_dim).
            view1 (torch.Tensor): The second view. Shape: (batch_size, embedding_dim).

        Returns:
            torch.FloatTensor: The first part of the contrastive loss.
        """
        view0 = torch.nn.functional.normalize(view0, p=2, dim=1)  # N x D
        view1 = torch.nn.functional.normalize(view1, p=2, dim=1)  # N x D

        cos01 = torch.matmul(view0, view1.t()) / self.temperature  # N x N
        positives = torch.diag(cos01)  # N
        nominator = torch.exp(positives)

        cos00 = torch.matmul(view0, view0.t()) / self.temperature  # N x N
        to_remove = torch.diag(cos00)

        cos = torch.cat([cos01, cos00], dim=1)

        if self.add_epsilon:
            m = view0.size(0)
            epsilon = self._compute_epsilon(m)
        else:
            epsilon = 0
        denominator = torch.sum(torch.exp(cos), dim=1) - \
            torch.exp(to_remove) + 2 * epsilon
        loss = -torch.log(nominator / denominator)

        return loss.mean() if self.reduction == "mean" else loss.sum()

    def _compute_epsilon(self, m):
        """
        Computes epsilon for the epsilon modified loss.

        Args:
            m (int): The batch size.

        Returns:
            float: Computed epsilon value.
        """
        delta = self.delta
        c_bound = np.exp(1 / self.temperature) - np.exp(-1 / self.temperature)
        epsilon = c_bound * \
            np.sqrt((2 * (m - 1) * np.log(2 / delta)) / (self.alpha))
        return epsilon


class SimplifiedContrastiveLoss:
    """
    Calculates the simplified contrastive loss (only m-1 negative samples).

    Args:
        temperature (float): Scaling factor for the similarity scores. Default is 0.1.
        add_epsilon (bool): Whether to compute the epsilon modified loss. Default is False.
        delta (float): A parameter used in epsilon computation (see Theorem 1). Default is 0.02.
        reduction (str): Options are 'mean' or 'sum'. Default is 'mean'.
        alpha (float): A parameter used in epsilon computation (see Theorem 1). Default is 0.1.

    Methods:
        forward(first_view, second_view):
            Calculates the final symmetric contrastive loss.

        contrastive_loss_single_view(view0, view1):
            Calculates the contrastive loss for a given order of parameters.

        _compute_epsilon(m):
            Computes epsilon for the epsilon-modified contrastive loss.
    """

    def __init__(
        self, temperature: float = 0.1, add_epsilon=False, delta=0.04, reduction="mean", alpha=0.1
    ):
        self.temperature = temperature
        self.add_epsilon = add_epsilon
        self.delta = delta
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, first_view: torch.Tensor, second_view: torch.Tensor) -> torch.FloatTensor:
        """
        Calculates the final contrastive loss for both views.

        Args:
            first_view (torch.Tensor): The first augmented view. Shape: (batch_size, embedding_dim).
            second_view (torch.Tensor): The second augmented view. Shape: (batch_size, embedding_dim).

        Returns:
            torch.FloatTensor: The contrastive loss.
        """
        first_loss = self.contrastive_loss_single_view(first_view, second_view)
        second_loss = self.contrastive_loss_single_view(
            second_view, first_view)
        losses = [first_loss, second_loss]
        final_loss = torch.stack(losses).mean()
        return final_loss

    def contrastive_loss_single_view(
        self, view0: torch.Tensor, view1: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Calculates the contrastive loss for a specific order of the views.

        Args:
            view0 (torch.Tensor): The first view. Shape: (batch_size, embedding_dim).
            view1 (torch.Tensor): The second view. Shape: (batch_size, embedding_dim).

        Returns:
            torch.FloatTensor: The first part of the contrastive loss.
        """
        view0 = torch.nn.functional.normalize(view0, p=2, dim=1)  # N x D
        view1 = torch.nn.functional.normalize(view1, p=2, dim=1)  # N x D

        cos01 = torch.matmul(view0, view1.t()) / self.temperature  # N x N
        positives = torch.diag(cos01)  # N
        nominator = torch.exp(positives)

        if self.add_epsilon:
            m = view0.size(0)
            epsilon = self._compute_epsilon(m)
        else:
            epsilon = 0
        denominator = torch.sum(torch.exp(cos01), dim=1) + 2 * epsilon
        loss = -torch.log(nominator / denominator)

        return loss.mean() if self.reduction == "mean" else loss.sum()

    def _compute_epsilon(self, m):
        """
        Computes epsilon for the epsilon modified loss.

        Args:
            m (int): The batch size.

        Returns:
            float: Computed epsilon value.
        """
        delta = self.delta
        c_bound = np.exp(1 / self.temperature) - np.exp(-1 / self.temperature)
        epsilon = c_bound * \
            np.sqrt(((m - 1) * np.log(2 / delta)) / (2 * self.alpha))
        return epsilon


class ZeroOneLoss:
    """
    Computes the Zero-One contrastive loss for two views of data.

    Args:
        reduction (str): Specifies the method for reducing the loss. Options are 'mean' or 'sum'. Default is 'mean'.

    Methods:
        forward(view0, view1):
            Calculates the Zero-One contrastive loss for the given pairs of augmented views.
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, view0: torch.Tensor, view1: torch.Tensor) -> torch.FloatTensor:
        """
        Calculates the Zero-One loss for the provided views.

        Args:
            view0 (torch.Tensor): The first augmented view. Shape: (batch_size, embedding_dim).
            view1 (torch.Tensor): The second augmented view. Shape: (batch_size, embedding_dim).

        Returns:
            torch.FloatTensor: The Zero-One Contrastive loss.
        """
        view0 = torch.nn.functional.normalize(view0, p=2, dim=1)  # N x D
        view1 = torch.nn.functional.normalize(view1, p=2, dim=1)  # N x D
        cos01 = torch.matmul(view0, view1.t())  # N x N
        positives = torch.diag(cos01)  # N
        size = cos01.shape[0]
        cos01[range(size), range(size)] = -float("inf")
        diff = positives - cos01
        loss = torch.sum(diff <= 0.0, dim=1).float() / (diff.shape[1] - 1)

        return loss.mean() if self.reduction == "mean" else loss.sum()
