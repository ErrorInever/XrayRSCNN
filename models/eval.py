import logging

logger = logging.getLogger(__name__)


def evaluate(model, dataloader, criterion, device, epoch, metric_logger, graph_loss, graph_acc, print_freq=100):
    """
    Evaluate one epoch
    :param model: instance of model
    :param dataloader:  dataloader
    :param criterion: loss function
    :param device: current device
    :param epoch: ``int`` number of epoch
    :param metric_logger: ``SummaryWriter``
    :param graph_loss: ``Session`` graph for loss metrics
    :param graph_acc: ``Session`` graph for accuracy metrics
    :param print_freq: ``int`` output frequency
    """
    logger.setLevel(logging.INFO)

    model.eval()

    epoch_loss = 0.
    epoch_acc = 0.

    running_loss = 0.
    running_acc = 0.

    for i, (image, label) in enumerate(dataloader):
        images = image.to(device)
        labels = label.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()
        running_acc += (outputs.argmax(dim=1) == labels).float().mean().item()

        epoch_loss += loss.item()
        epoch_acc += (outputs.argmax(dim=1) == labels).float().mean().item()

        if (i % print_freq == 0) and i != 0:
            logger.info('Evaluate [%s, %s] Loss: %s | Acc: %s', epoch + 1, i + 1,
                        running_loss / print_freq, running_acc / print_freq)
            running_loss = 0.
            running_acc = 0.

    logger.info('----------EVAL EPOCH [%s] | LOSS: %s | EPOCH ACC: %s ---------', epoch + 1,
                epoch_loss / len(dataloader), epoch_acc / len(dataloader))
    metric_logger.add_scalar('loss/eval', epoch_loss / len(dataloader), epoch)
    metric_logger.add_scalar('acc/eval', epoch_acc / len(dataloader), epoch)
    graph_loss.append(epoch, {'val_loss': epoch_loss / len(dataloader)})
    graph_acc.append(epoch, {'val_acc': epoch_acc / len(dataloader)})
    metric_logger.close()
