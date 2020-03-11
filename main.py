import os
import argparse
import logging
import time
import torch
from config.conf import cfg
from data.dataset import XRayDataset
from utils.parse import get_data_frame
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='X-ray-CNN')
    parser.add_argument('--root_dir', dest='root_dir', help='Path to root directory of dataset', default=None, type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', action='store_true')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
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

    logger.setLevel(logging.DEBUG)

    args = parse_args()

    logger.info('XrayPneumoniaCNN starts training {}'.format(time.ctime()))
    logger.info('Called with args: {}'.format(args.__dict__))
    logger.info('Config params:{}'.format(cfg.__dict__))

    if torch.cuda.is_available() and not args.use_gpu:
        logger.info('You have a GPU device so you should probably run with --use_gpu')
        device = torch.device('cpu')
    elif torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    logger.info('Running with device %s', device)
    logger.info('Creates datasets')

    train_df = get_data_frame(os.path.join(args.root_dir, 'train'))
    test_df = get_data_frame(os.path.join(args.root_dir, 'test'))
    val_df = get_data_frame(os.path.join(args.root_dir, 'val'))

    train_dataset = XRayDataset(train_df, transforms=True)
    test_df = XRayDataset(test_df)
    val_df = XRayDataset(val_df)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_df, batch_size=cfg.BATCH_SIZE)
    val_dataloader = DataLoader(val_df, batch_size=cfg.BATCH_SIZE)
    logger.info('Datasets created: Train size %s Test size %s Val size %s', len(train_dataloader),
                len(test_dataloader), len(val_dataloader))
