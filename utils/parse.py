import pandas as pd
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_data_frame(root_dir):
    """
    Creates table
    :param root_dir: root directory of dataset
    :return: ``DataFrame`` [img_path, label]
    """
    df = pd.DataFrame(columns=['img_path', 'label'])
    for cls in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls)
        for img_name in tqdm(os.listdir(cls_path)):
            if img_name.endswith(('.jpeg', 'jpg', '.png')):
                df = df.append(pd.DataFrame({'img_path': [os.path.join(cls_path, img_name)],
                                             'label': [cls]}), ignore_index=True)
    return df
