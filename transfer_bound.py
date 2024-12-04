import math
import numpy as np
import torch
from sklearn.preprocessing import normalize


class Bound:
    """
    Class to compute an upper-bound on the linear classifier loss based on contrastive loss using 
    either (1) bound from Bao et al. (2) Theorem 3.

    Attributes:
        tau (float): Temperature parameter used in the calculation of delta.
        num_neg_samples (int): Number of negative samples used in the delta calculation.
        num_classes (int): Number of classes in the dataset.
    """

    def __init__(self, tau=1, num_neg_samples=250, num_classes=10):
        super().__init__()
        self.tau = tau
        self.num_neg_samples = num_neg_samples
        self.num_classes = num_classes

    def calculate_delta(self, index=1):
        if index == 1:
            delta = math.log(
                self.num_classes / self.num_neg_samples *
                (math.cosh(1 / self.tau) ** 2)
            )
        elif index == 2:
            term1 = math.log(math.cosh(1) ** 2)
            term2 = self.tau * math.log(
                self.num_classes / self.num_neg_samples *
                (math.cosh(1 / self.tau) ** 2)
            )
            delta = min(term1, term2) + math.log(self.num_classes)

        return delta

    def forward(self, contrastive_loss, sigma, index):
        delta = self.calculate_delta(index=index)
        if index == 1:
            bound = contrastive_loss + sigma / self.tau + delta
        elif index == 2:
            bound = self.tau * contrastive_loss + sigma + delta
        return bound


class Sigma:
    """
    Class to compute the sigma value, which is the average feature deviation.

    Attributes:
        num_classes (int): Number of classes in the dataset.
        device (torch.device): The device to perform computations on.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _compute_cluster_means(self, embeddings, y):
        cluster_means = []
        for cluster in range(self.num_classes):
            cluster_points = embeddings[y == cluster]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_means.append(cluster_mean)
        return np.array(cluster_means)

    def _compute_l2_distance(self, embeddings, y, cluster_means):
        l2_distances = []
        for cluster in range(self.num_classes):
            cluster_points = embeddings[y == cluster]
            distances = np.linalg.norm(
                cluster_points - cluster_means[cluster], axis=1)
            l2_distances.extend(distances.tolist())
        return np.mean(l2_distances)

    def forward(self, model, train_loader):
        embeddings = []
        y = []
        for images, labels in train_loader:
            images = images.to(self.device)
            outputs = model(images, sample=True)
            embeddings.extend(outputs.tolist())
            y.extend(labels.tolist())
        x_normalized = normalize(embeddings, norm="l2")
        y = np.array(y)
        cluster_means = self._compute_cluster_means(x_normalized, y)
        l2_distance = self._compute_l2_distance(x_normalized, y, cluster_means)
        return l2_distance
