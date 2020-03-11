from easydict import EasyDict as edict

__C = edict()
# for consumers
cfg = __C

__C.PATH_TO_LOG_FILE = r'logs/logs.log'

# TRAIN

__C.NUM_EPOCHS = 20
__C.BATCH_SIZE = 5
