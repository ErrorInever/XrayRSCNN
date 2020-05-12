from easydict import EasyDict as edict

__C = edict()
# for consumers
cfg = __C

__C.PATH_TO_LOG_FILE = r'XrayRSCNN/logs/logs.log'
__C.OUT_DIR = 'out'
# TRAIN
__C.NUM_EPOCHS = 15
__C.BATCH_SIZE = 5
__C.LEARNING_RATE = 1e-2

# SGD
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 0.001

# SCHEDULER STEP LR
__C.STEP_SIZE = 8
__C.GAMMA = 0.1

__C.SAVE_EPOCH_NUM = 75
