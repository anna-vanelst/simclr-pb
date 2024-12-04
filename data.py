from torchvision import transforms, datasets
from torch.utils.data import Dataset


def load_train_test(name="cifar10", transform=None):
    """
    Loads the training and test datasets for the specified dataset.

    Args:
        name (str): The name of the dataset to load. Supported values are 'mnist' and 'cifar10'.
                    Default is 'cifar10'.
        transform (transforms.Compose, optional): The transformation to apply to the dataset. Default is None.

    Returns:
        A tuple containing the training and test datasets.
    """
    if name == "mnist":
        train = datasets.MNIST("mnist-data/", train=True,
                               download=True, transform=transform)
        test = datasets.MNIST("mnist-data/", train=False,
                              download=True, transform=transform)
    elif name == "cifar10":
        train = datasets.CIFAR10(
            "cifar-data/", train=True, download=True, transform=transform)
        test = datasets.CIFAR10(
            "cifar-data/", train=False, download=True, transform=transform)
    else:
        raise RuntimeError(
            f"Invalid data name: {name}. Supported values: 'mnist','cifar10'")
    return train, test


def data_transform(data_name="cifar10") -> transforms.Compose:
    """
    Returns a data transformation pipeline for the specified dataset.

    Args:
        data_name (str): The name of the dataset for which the transformation is created. 
        Supported values are 'mnist' and 'cifar10'. Default is 'cifar10'.

    Returns:
        transforms.Compose: A composed transform with data normalization.
    """
    if data_name == "mnist":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif data_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std_dev = (0.2470, 0.2435, 0.2616)
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std_dev)])
    else:
        raise ValueError(
            f"Invalid data name: {data_name}. Supported values: 'mnist','cifar10'")


def create_simclr_data_augmentation(strength=0.5, name="cifar10") -> transforms.Compose:
    """
    Creates a data augmentation pipeline for SimCLR.

    Args:
        strength (float): Controls the strength of color jittering. Default is 0.5.
        name (str): The dataset name for which the augmentation is created. Supported values are 
        'mnist' and 'cifar10'. Default is 'cifar10'.

    Returns:
        transforms.Compose: A composed transform with a series of data augmentation techniques.
    """
    if name == "mnist":
        size = 28
        mean = (0.1307,)
        std_dev = (0.3081,)
    elif name == "cifar10":
        size = 32
        scale = (0.08, 1.0)
        mean = (0.4914, 0.4822, 0.4465)
        std_dev = (0.2470, 0.2435, 0.2616)
    else:
        raise ValueError(
            f"Invalid data name: {name}. Supported values: 'mnist','cifar10'")

    scale = (0.08, 1.0)
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * strength,
        contrast=0.8 * strength,
        saturation=0.8 * strength,
        hue=0.2 * strength,
    )

    common_transforms = [
        transforms.RandomResizedCrop(size=size, scale=scale),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(transforms=[color_jitter], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std_dev)
    ]

    return transforms.Compose(common_transforms)


class SimCLRAugmentedDataset(Dataset):
    """
    A dataset wrapper that applies SimCLR data augmentation to each image in the dataset.

    Args:
        dataset (Dataset): The original dataset to augment.
        name (str): The dataset name to specify the augmentation parameters. Default is 'cifar10'.

    Attributes:
        data (Dataset): The original dataset.
        transform (transforms.Compose): The SimCLR data augmentation pipeline.
    """

    def __init__(self, dataset, name="cifar10"):
        self.data = dataset
        self.transform = create_simclr_data_augmentation(
            strength=0.5, name=name)

    def __getitem__(self, index):
        img, _ = self.data[index]
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.data)
