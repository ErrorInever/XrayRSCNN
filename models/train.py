import torch
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm


def train_one_epoch(model, loss, optimizer, data_loader,
                    device, out_dir, tensorboard=False):
    """
    :param model:
    :param loss:
    :param optimizer:
    :param data_loader:
    :param device:
    :param out_dir:
    :param running_loss:
    :param running_acc:
    :param tensorboard:
    :param print_freq:
    :return:
    """
    if tensorboard:
        logger = SummaryWriter(out_dir)

    running_loss = 0.
    running_acc = 0.

    for images, labels in tqdm(data_loader, total=len(data_loader)):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            predicts = model(images)
            loss_value = loss(predicts, labels)
            predicts_class = predicts.argmax(dim=1)
            loss_value.backward()
            optimizer.step()

        running_loss += loss_value.item()
        running_acc += (predicts_class == labels.data).float().mean()

    return running_loss, running_acc
