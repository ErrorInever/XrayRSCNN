

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

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        predicts = model.forward(images)

        # if labels.dim() == 1 after squeeze() tensor will not have dimension
        try:
            loss_value = loss(predicts, labels.squeeze())
        except IndexError:
            loss_value = loss(predicts, labels.squeeze(0))
        finally:
            running_loss += loss_value.item()
            running_acc += (predicts.argmax() == labels.data).float().mean()

    return running_loss, running_acc

