import logging

logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch,
                    metric_logger, graph_loss, graph_acc, print_freq=100):
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

    logger.info('---------- EPOCH [%s] | LOSS: %s | EPOCH ACC: %s ---------', epoch + 1,
                epoch_loss / len(dataloader), epoch_acc / len(dataloader))

    metric_logger.add_scalar('loss/train', epoch_loss / len(dataloader), epoch)
    metric_logger.add_scalar('acc/train', epoch_acc / len(dataloader), epoch)
    graph_loss.append(epoch, {'train_loss': epoch_loss / len(dataloader)})
    graph_acc.append(epoch, {'train_acc': epoch_acc / len(dataloader)})
    metric_logger.close()
