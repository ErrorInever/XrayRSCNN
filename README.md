Neural network for detecting pneumonia

### Works on python 3.7 and lower

### Install
1. > clone this repo ``git clone https://github.com/ErrorInever/XrayRSCNN``
2. > install requirements: ``pip install -r requirements.txt``

### Start inference

    optional arguments:
      -h, --help                    show this help message and exit
      --weight_path WEIGHT_PATH     Path to directory where weights of model stored
      --use_gpu                     use gpu
      --test                        get confusion matrix, f-score
      --test_data TEST_DATA         Path to test images folder
      --inference                   inference mode
      --img IMG_PATH                Path to image
                                 

### example run
 > python inference.py --weight_path "models/weights/xray.pth" --inference --img "folder/pneumo.png" --use_gpu

![alt text](https://raw.githubusercontent.com/ErrorInever/XrayRSCNN/master/images/normal_true.png)

### Metrics pre-trained model
        Accuracy: 88.18%
        Recall: 99.32%
        Precision: 79.35%
        F1-SCORE 88.22%

![alt text](https://raw.githubusercontent.com/ErrorInever/XrayRSCNN/master/images/confusion_matrix.png)


### Start train

    optional arguments:
      -h, --help           show this help message and exit
      --root_dir ROOT_DIR  Path to root directory of dataset
      --use_gpu            use gpu
      --api_key API_KEY    losswise api key
      --out_dir OUT_DIR    Path to out directory
      --save_model         save model

# example run
 > python train_val.py --root_dir "path/to/data/dir" --outdir "path/to/output/dir" --use_gpu --save_model

# you can edit **config/conf.py** for change params
        __C.NUM_EPOCHS = 15
        __C.BATCH_SIZE = 5
        __C.LEARNING_RATE = 1e-2

        # SGD
        __C.MOMENTUM = 0.9
        __C.WEIGHT_DECAY = 0.001

        # SCHEDULER STEP LR
        __C.STEP_SIZE = 8
        __C.GAMMA = 0.1

        # ReduceLROnPlateau
        __C.PATIENCE = 3
        __C.VERBOSE = True

        __C.SAVE_EPOCH_NUM = 75

