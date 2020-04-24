import logging

logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch, metric_logger, print_freq=100):
    logger.setLevel(logging.INFO)

    epoch_loss = 0.
    epoch_acc = 0.

    running_loss = 0.
    running_acc = 0.

    model.train()
    for i, (image, label) in enumerate(dataloader):
        images = image.to(device)
        labels = label.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()

        optimizer.step()
        scheduler.step()

        # statistics
        running_loss += loss.item()
        running_acc += (outputs.argmax(dim=1) == labels.data).float().mean().item()

        epoch_loss += loss.item()
        epoch_acc += (outputs.argmax(dim=1) == labels.data).float().mean().item()

        if (i % print_freq == 0) and i != 0:
            logger.info('Train [%s, %s] Loss: %s | Acc: %s', epoch + 1, i + 1,
                        running_loss / print_freq, running_acc / print_freq)

            running_loss = 0.
            running_acc = 0.

    logger.info('---------- EPOCH LOSS: %s | EPOCH ACC: %s ---------',
                epoch_loss / len(dataloader), epoch_acc / len(dataloader))

    metric_logger.add_scalar('train/loss', epoch_loss, epoch)
    metric_logger.add_scalar('train/acc', epoch_acc, epoch)
