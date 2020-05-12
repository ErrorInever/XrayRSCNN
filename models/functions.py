import torch
import os
import random
import numpy as np
import logging
from torch import nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(model, filepath):
    """load model for test"""
    logger.info('Loading model...')
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


def save_model(model, filepath):
    """save model for test"""
    torch.save(model.state_dict(), os.path.join(filepath, 'xray.pth'))


def set_seed(val):
    """freeze random sequences"""
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.backends.cudnn.deterministic = True


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=False)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=False)],
        ['selu', nn.SELU(inplace=False)],
        ['none', nn.Identity()]
    ])[activation]


def save_checkpoint(state, filename):
    """save current state of model"""
    torch.save(state, filename)


def make_checkpoint(epoch, model, optimizer, scheduler, loss_value, outdir):
    save_name = os.path.join(outdir, 'xrayscnn_ep{}.pth'.format(epoch))
    save_checkpoint({
        'start_epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'losses': loss_value
    }, save_name)
    print('Save model: {}'.format(save_name))


def load_checkpoint(state, model, optimizer, scheduler):
    """load previous state of model"""
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])