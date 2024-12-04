import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


class LinearClassifier(nn.Module):
    """
    A linear classifier for feature representations learned by a SimCLR model.

    Args:
        simclr_model (nn.Module): A SimCLR model used to extract features.
        num_classes (int): Number of output classes for classification. Default is 10.
        projection (bool): Whether to use projection head in SimCLR. Default is False.
        data_name (str): Name of the dataset ('mnist' or 'cifar10') to determine feature size. Default is 'cifar10'.
    """

    def __init__(self, simclr_model, num_classes=10, projection=False, data_name="cifar10"):
        super(LinearClassifier, self).__init__()
        self.simclr_model = simclr_model
        for param in self.simclr_model.parameters():
            param.requires_grad = False
        self.projection = projection
        if self.projection:
            SimCLR_FEATURE_SIZE = 128
        else:
            if data_name == "mnist":
                SimCLR_FEATURE_SIZE = 512
            elif data_name == "cifar10":
                SimCLR_FEATURE_SIZE = 2048
            else:
                raise ValueError(
                    "Unsupported data_name. Choose 'mnist' or 'cifar10'.")
        self.linear_layer = nn.Linear(
            in_features=SimCLR_FEATURE_SIZE, out_features=num_classes)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, sample=True):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.
            sample (bool): Whether to use stochastic network. Default is True.

        Returns:
            torch.Tensor: Output logits.
        """
        with torch.no_grad():
            representations = self.simclr_model(
                x, sample=sample, projection=self.projection)
        logits = self.linear_layer(representations)
        return logits

    def train_classifier(self, train_loader, num_epochs=20, lr=0.01, verbose=False):
        """
        Train the classifier on the given training data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train. Default is 20.
            lr (float): Learning rate for the optimizer. Default is 0.01.
        """
        device = self.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.train()
            losses = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if verbose:
                print(f"Epoch {epoch}: {np.mean(losses):.4f}")
        supervised_train_loss = np.mean(losses)
        print(f"Supervised train loss: {supervised_train_loss:.4f}")

    def test_classifier(self, test_loader, verbose=False):
        """
        Evaluate the classifier on the given test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            None
        """
        device = self.device
        self.eval()
        predictions = []
        true_labels = []
        criterion = nn.CrossEntropyLoss()
        losses = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                loss = criterion(outputs, labels)
                losses.append(loss.item())
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Supervised test loss: {np.mean(losses):.4f}")
        print(f"Accuracy: {accuracy:.4f}")
