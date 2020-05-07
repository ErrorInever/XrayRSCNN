import logging
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)


def test(model, dataloader, device):
    logger.setLevel(logging.INFO)
    logger.info('--------START TEST MODEL---------')
    model.eval()
    test_acc = 0.

    true_labels = []
    pred_labels = []

    for images, labels in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predict_classes = outputs.argmax(dim=1)
        test_acc += (predict_classes == labels.data).float().mean().item()

        true_labels.append(labels.detach().cpu())
        pred_labels.append(predict_classes.detach().cpu())

    test_acc = test_acc / len(dataloader)

    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * (recall * precision) / (precision + recall)

    logger.info('Test accuracy: %s', round(test_acc, 2))
    logger.info('Recall: %s | Precision: %s | F1_SCORE %s', recall, precision, f1_score)

    svm = sns.heatmap(cm, annot=True, fmt="d")
    fig = svm.get_figure()
    fig.savefig('heatmap.png', dpi=400)

