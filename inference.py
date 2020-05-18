import logging
import argparse
import torch
import torchvision
import cv2
from datetime import datetime
from models.model import XrayMRSCNN
from models.functions import load_model
from config.conf import cfg
from data.dataset import XrayImageFolder
from torch.utils.data import DataLoader
from models.test import test


def parse_args():
    parser = argparse.ArgumentParser(description='X-ray-RCNN')
    parser.add_argument('--weight_path', dest='weight_path', help='Path to directory where weights of model stored',
                        default=None, type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', action='store_true')
    parser.add_argument('--test', dest='test', help='get confusion matrix, f-score', action='store_true')
    parser.add_argument('--test_data', dest='test_data', help='Path to test images folder', default=None, type=str)
    parser.add_argument('--inference', dest='inference', help='inference mode', action='store_true')
    parser.add_argument('--img', dest='img_path', help='Path to image', default=None,
                        type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.weight, 'weight path not specified'
    if args.inference:
        assert args.img_path, 'image path not specified'
    if args.test:
        assert args.test_data, 'test data path not specified'
    if args.inference is None and args.test is None:
        raise NameError('arguments --inference and --test - not specified')

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

    logger.info('Called with args: {}'.format(args.__dict__))
    logger.info('Config params:{}'.format(cfg.__dict__))

    model = XrayMRSCNN()
    model = load_model(model, args.weight_path)

    if torch.cuda.is_available() and not args.use_gpu:
        logger.info('You have a GPU device so you should probably run with --use_gpu')
        device = torch.device('cpu')
    elif torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model.to(device)
    logger.info('Running with device %s', device)

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    if args.test:
        test_dataset = XrayImageFolder(args.test_data, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)
        test(model, test_dataloader, device)

    elif args.inference:
        img = cv2.imread(args.img_path)
        img = transform(img).unsquuze(0)

        start_time = datetime.now()
        with torch.no_grad():
            output = model.inference(img)

        label = output.data.argmax()
        prob = output.detach()[0][label].item() * 100
        pred_class = 'NORMAL' if label.item() == 0 else 'PNEUMONIA'

        logger.info('Detection finished in %s seconds', (datetime.now() - start_time).total_seconds())
        logger.info('Predicted class: %s\nProbability%s%', pred_class, round(prob, 1))
