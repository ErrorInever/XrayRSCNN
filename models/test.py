import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def test(model, dataloader, device):
    logger.setLevel(logging.INFO)
    logger.info('--------START TEST MODEL---------')
    model.eval()
    test_acc = 0.
    for images, labels in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predict_classes = outputs.argmax(dim=1)
        test_acc += (predict_classes == labels.data).float().mean()

    test_acc = test_acc / len(dataloader)

    logger.info('Test accuracy: %s', test_acc)
