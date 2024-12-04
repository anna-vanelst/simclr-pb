import torch


def evaluate_contrastive_loss(prob_net, data_loader, contrastive_loss, device, sample=True) -> float:
    """
    Computes the average contrastive loss over a dataset using a given probabilistic network.

    Args:
        prob_net: A neural network model that takes input data and outputs features.
        data_loader: DataLoader providing batches of input data for evaluation.
        contrastive_loss: Either ZeroOneLoss, SimplifiedContrastiveLoss, ContrastiveLoss
        device: The device (e.g., 'cpu' or 'cuda') on which to perform computations.
        sample: A boolean indicating whether to use a stochastic network.

    Returns:
        float: The average contrastive loss computed over the dataset.
    """
    prob_net.eval()
    avgloss = 0.0
    with torch.no_grad():
        for batch_id, (view0, view1) in enumerate(data_loader):
            list_views = [view0, view1]
            list_views = [data.to(device) for data in list_views]
            features = [prob_net(data, sample=sample) for data in list_views]
            loss = contrastive_loss.forward(features[0], features[1])
            avgloss += loss.item()
        avgloss = avgloss / batch_id
    return avgloss
