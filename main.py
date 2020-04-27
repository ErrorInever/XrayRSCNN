import os
import argparse
import logging
import time
import torch
import datetime
import losswise
import kornia
from config.conf import cfg
from data.dataset import XRayDataset
from utils.parse import get_data_frame
from torch.utils.data import DataLoader
from models.model import XraySCNN, XrayCNN
from models.train import train_one_epoch
from models.eval import evaluate
from models.test import test
from models import functions
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='X-ray-CNN')
    parser.add_argument('--root_dir', dest='root_dir', help='Path to root directory of dataset', default=None, type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', action='store_true')
    parser.add_argument('--api_key', dest='api_key', help='losswise api key', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path to out directory', default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    functions.set_seed(10)

    args = parse_args()
    assert args.root_dir, 'Root directory not specified'
    assert args.out_dir, 'Out directory not specified'

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

    logger.info('XrayPneumoniaCNN starts training {}'.format(time.ctime()))
    logger.info('Called with args: {}'.format(args.__dict__))
    logger.info('Config params:{}'.format(cfg.__dict__))

    losswise.set_api_key(args.api_key)
    session = losswise.Session(tag='x-ray-test',
                               params={'Adam learning rate': cfg.LEARNING_RATE,
                                       'Scheduler gamma': 0.1},
                               max_iter=cfg.NUM_EPOCHS,
                               track_git=False)

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
    test_dataset = XRayDataset(test_df)
    val_dataset = XRayDataset(val_df)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    logger.info('Datasets created: Train batches %s Test batches %s Val batches %s', len(train_dataloader),
                len(test_dataloader), len(val_dataloader))

    model = XraySCNN()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.8, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    metric_logger = SummaryWriter()
    graph_loss = session.graph('loss', kind='min', display_interval=1)
    graph_accuracy = session.graph('accuracy', kind='max', display_interval=1)

    # ---- Train model -----
    start_time = time.time()

    for epoch in range(cfg.NUM_EPOCHS):
        train_one_epoch(model, train_dataloader, optimizer, criterion, scheduler,
                        device, epoch, metric_logger, graph_loss, graph_accuracy, args.out_dir, print_freq=100)
        evaluate(model, val_dataloader, criterion, device, epoch, metric_logger, graph_loss,
                 graph_accuracy, print_freq=5)
    test(model, test_dataloader, device)
    functions.save_model(model, args.out_dir)

    session.done()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training ended')
    logger.info('Training time %s', total_time_str)
