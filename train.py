# Credit: M. Perez-Ortiz - https://github.com/mperezortiz/PBB
from loss import ZeroOneLoss, SimplifiedContrastiveLoss


def trainNNet(net, optimizer, epoch, train_loader, temperature, device, verbose=True):
    """
    Trains a neural network for one epoch using a contrastive learning approach.

    Args:
        net (torch.nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        epoch (int): The current epoch number.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        temperature (float): Temperature parameter for the contrastive loss. 
        device (torch.device): The device to run the training on ('cuda' or 'cpu').
        verbose (bool, optional): Whether to print training progress. Default is True.

    Returns:
        tuple: A tuple containing the average training loss and the average zero-one error across all batches.
    """
    net.train()
    avg_0_1_loss, avgloss = 0.0, 0.0
    for batch_id, (view0, view1) in enumerate(train_loader):
        list_views = [view0, view1]
        list_views = [data.to(device) for data in list_views]
        net.zero_grad()
        features = [net(data) for data in list_views]
        contrastive_loss = SimplifiedContrastiveLoss(temperature=temperature)
        loss = contrastive_loss.forward(features[0], features[1])
        loss.backward()
        optimizer.step()
        zero_one_loss = ZeroOneLoss()
        loss_0_1 = zero_one_loss.forward(features[0], features[1])
        avg_0_1_loss += loss_0_1.item()
        avgloss += loss.item()
    if verbose:
        print(
            f"-Epoch {epoch}, Train loss: {avgloss/batch_id :.4f}, Train err:  {avg_0_1_loss/batch_id :.4f}")


def trainPNNet(net, optimizer, pbobj, epoch, train_loader, verbose=True):
    """
    Trains a probabilistic neural network for one epoch using PAC-Bayes by backprop.

    Args:
        net (torch.nn.Module): The probabilistic neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        pbobj: An object that encapsulates the probabilistic bound and training objective.
        epoch (int): The current epoch number.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        verbose (bool, optional): Whether to print training progress. Default is True.

    Returns:
        tuple: A tuple containing the average bound, average KL divergence, average contrastive loss,
               and average zero-one error across all batches.
    """
    net.train()
    avgerr, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0

    for batch_id, (view0, view1) in enumerate(train_loader):
        list_views = [view0, view1]
        list_views = [data.to(pbobj.device) for data in list_views]
        net.zero_grad()
        features = [net(data, sample=True) for data in list_views]
        bound, kl_n, loss, err = pbobj.train_obj(net, features)
        bound.backward()
        optimizer.step()
        avgbound += bound.item()
        avgkl += kl_n
        avgloss += loss.item()
        avgerr += err.item()

    if verbose:
        print(f"-Batch average epoch {epoch} results, Train obj: {avgbound/batch_id :.4f}, KL/n: {avgkl/batch_id :.4f}, Contrastive loss: {avgloss/batch_id :.4f}, Train 0-1 Error:  {avgerr/batch_id :.4f}")
