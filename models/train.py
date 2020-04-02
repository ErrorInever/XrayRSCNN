import torch
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)


def train_one_epoch(model, loss, optimizer, data_loader, device):
    """
    :param model:
    :param loss:
    :param optimizer:
    :param data_loader:
    :param device:
    :return:
    """
    logger.setLevel(logging.INFO)

    running_loss = 0.
    running_acc = 0.

    for images, labels in tqdm(data_loader, total=len(data_loader)):

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            predicts = model(images)
            loss_value = loss(predicts, labels.squeeze())

            loss_value.backward()
            optimizer.step()

        running_loss += loss_value.item()
        running_acc += (predicts.argmax(dim=1) == labels.data).float().mean()

    return running_loss, running_acc
