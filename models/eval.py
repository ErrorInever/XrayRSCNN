from tqdm import tqdm


def evaluate(model, dataloader, loss, device):
    """
    :param model:
    :param dataloader:
    :param loss:
    :param device:
    :return:
    """
    running_loss = 0.
    running_acc = 0.

    for images, labels in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        predicts = model.forward(images)

        loss_value = loss(predicts, labels.squeeze())

        running_loss += loss_value.item()
        running_acc += (predicts.argmax() == labels.data).float().mean()

    return running_loss, running_acc

