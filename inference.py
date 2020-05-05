import logging
import argparse
import torch
from models.model import XrayRSCNN
from models.functions import load_model
from config.conf import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='X-ray-CNN')
    parser.add_argument('--weight_path', dest='weight', help='Path to directory where weights of model stored',
                        default=None, type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', action='store_true')
    parser.add_argument('--get_metrics', dest='get_metrics', help='get confusion matrix', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.weight, 'weight path not specified'

    logger = logging.getLogger()

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(cfg.PATH_TO_LOG_FILE, mode='w', encoding='utf-8')

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.ERROR)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.setLevel(logging.INFO)

    model = XrayRSCNN()
    model = load_model(model, args.weight)

    if torch.cuda.is_available() and not args.use_gpu:
        logger.info('You have a GPU device so you should probably run with --use_gpu')
        device = torch.device('cpu')
    elif torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    logger.info('Running with device %s', device)

    if args.get_metrics:
        pass