import os
import argparse
import logging
import time
import torch
import datetime
import losswise
import torchvision
from torchvision import transforms
from config.conf import cfg
from data.dataset import XrayImageFolder
from torch.utils.data import DataLoader
from models.model import XrayMRSCNN
from models.train import train_one_epoch
from models.eval import evaluate
from models.test import test
from models import functions
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='X-ray-RCNN')
    parser.add_argument('--root_dir', dest='root_dir', help='Path to root directory of dataset', default=None, type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', action='store_true')
    parser.add_argument('--api_key', dest='api_key', help='losswise api key', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path to out directory', default=None, type=str)
    parser.add_argument('--save_model', dest='save_model', help='save model', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    functions.set_seed(42)

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

    train_transform = transforms.Compose([
        transforms.RandomAffine(0, translate=(0, 0.1), scale=(1, 1.10)),
        transforms.RandomRotation((-20, 20)),
        transforms.ToTensor(),
    ])

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    train_dataset = XrayImageFolder(os.path.join(args.root_dir, 'train'), transform=train_transform)
    val_dataset = XrayImageFolder(os.path.join(args.root_dir, 'val'), transform=transform)
    test_dataset = XrayImageFolder(os.path.join(args.root_dir, 'test'), transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    logger.info('Datasets created: Train batches %s Test batches %s Val batches %s', len(train_dataloader),
                len(test_dataloader), len(val_dataloader))

    model = XrayMRSCNN()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.PATIENCE, verbose=cfg.VERBOSE)

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

    if args.save_model:
        functions.save_model(model, args.out_dir)

    session.done()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training ended')
    logger.info('Training time %s', total_time_str)
