import torch
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm
logger = logging.getLogger(__name__)


def train_one_epoch(model, loss, optimizer, data_loader,
                    device, out_dir, epoch, tensorboard=True, print_freq=30):
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
    logger.setLevel(logging.INFO)

    if tensorboard:
        metric_logger = SummaryWriter(out_dir)

    running_loss = 0.
    running_acc = 0.
    freq_value = 0
    step_batch = 0

    for images, labels in tqdm(data_loader, total=len(data_loader)):
        freq_value += 1
        step_batch += 1

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

        if tensorboard:
            metric_logger.add_scalar('train/losses', running_loss / len(data_loader), step_batch)
            metric_logger.add_scalar('train/acc', running_acc / len(data_loader), step_batch)
            metric_logger.close()

        if freq_value % print_freq == 0:
            logger.info('[Running Loss: {:.4f} | Running Acc: {:.4f}]'.format(running_loss / len(data_loader),
                                                                              running_acc / len(data_loader)))

    return running_loss, running_acc
