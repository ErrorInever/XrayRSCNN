import torch
import seaborn as sns
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)


def get_cls_metrics(true_labels, pred_labels):
    """
    Calculation of metrics: Recall, Precision, F1-score
    :param true_labels: ``Tensor`` true labels
    :param pred_labels: ``Tensor`` predicted labels
    :return:
    if not exceptions
    ``float`` Recall metric,
    ``float`` Precision metric,
    ``float`` F1-score metric
    else
    ``None``
    ``None``
    ``None``
    """
    true_labels = torch.cat(true_labels).numpy()
    pred_labels = torch.cat(pred_labels).numpy()

    cm = confusion_matrix(true_labels, pred_labels)

    try:
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    finally:
        svm = sns.heatmap(cm, annot=True, fmt="d", cmap='coolwarm', linecolor='black', linewidths=1, cbar=False,
                          xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
        svm.set(title="Confusion matrix")
        fig = svm.get_figure()
        fig.savefig('confusion_matrix.png', dpi=400)

    try:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * (recall * precision) / (precision + recall)

    except ZeroDivisionError:
        logger.exception('ZeroDivisionError, tn:%s, fp:%s, fn:%s, tp:%s', tn, fp, fn, tp)
        recall = None
        precision = None
        f1_score = None

    return recall, precision, f1_score


def test(model, dataloader, device):
    """
    Tests model on data
    :param model: instance of model
    :param dataloader: instance of dataloader
    :param device: current device
    """
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

    recall, precision, f1_score = get_cls_metrics(true_labels, pred_labels)

    logger.info('Accuracy: %s\nRecall: %s\nPrecision: %s\nF1-SCORE %s\n', round(100 * test_acc, 2),
                round(100 * recall, 2), round(100 * precision, 2), round(100 * f1_score, 2))
