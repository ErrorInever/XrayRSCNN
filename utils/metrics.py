import logging

logger = logging.getLogger(__name__)


def print_metrics(name, epoch_num, metric_logger, loss, acc, length):
    logger.setLevel(logging.INFO)
    loss = loss / length
    acc = acc / length
    logger.info('{} [Epoch Loss: {:.4f} | Epoch Acc: {:.4f}]'.format(name, loss, acc))
    metric_logger.add_scalar('{}/loss'.format(name), loss, epoch_num)
    metric_logger.add_scalar('{}/acc'.format(name), acc, epoch_num)
