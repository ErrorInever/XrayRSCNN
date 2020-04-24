import logging

logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch, print_freq=100):
    logger.setLevel(logging.INFO)

    running_loss = 0.
    running_acc = 0.

    model.train()
    for i, image, label in enumerate(dataloader, 0):
        images = image.to(device)
        labels = label.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        # statistics
        running_loss += loss.item()
        running_acc += (outputs.argmax(dim=1) == labels.data).float().mean()

        if i % print_freq == 0:
            logger.info('[%s, %s] Loss: %s | Acc: %s', epoch + 1, i + 1,
                        running_loss / print_freq, running_acc / print_freq)
            running_loss = 0.
            running_acc = 0.
