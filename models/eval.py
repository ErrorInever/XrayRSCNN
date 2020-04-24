import logging

logger = logging.getLogger(__name__)


def evaluate(model, dataloader, criterion, device, epoch, metric_logger, print_freq=100):
    logger.setLevel(logging.INFO)

    model.eval()

    running_loss = 0.
    running_acc = 0.

    for i, (image, label) in enumerate(dataloader):
        images = image.to(device)
        labels = label.to(device)

        outputs = model(images)
        try:
            loss = criterion(outputs, labels.squeeze())
        except IndexError:
            loss = criterion(outputs, labels.squeeze(0))
        finally:
            running_loss += loss.item()
            running_acc += (outputs.argmax(dim=1) == labels).float().mean().item()

            if (i % print_freq == 0) and i != 0:
                logger.info('Evaluate [%s, %s] Loss: %s | Acc: %s', epoch + 1, i + 1,
                            running_loss / print_freq, running_acc / print_freq)
                running_loss = 0.
                running_acc = 0.
