from easydict import EasyDict as edict

__C = edict()
# for consumers
cfg = __C

__C.PATH_TO_LOG_FILE = r'XrayRSCNN/logs/logs.log'
__C.OUT_DIR = 'out'
# TRAIN

__C.NUM_EPOCHS = 5
__C.BATCH_SIZE = 5
__C.LEARNING_RATE = 0.01
