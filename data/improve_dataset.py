import os
import collections
import imgaug.augmenters as iaa
import cv2
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def horizontal_flip_all_images(root_dir, output_dir):
    """
    Horizontal flip all images from specified directory, copy and save them to output directory
    :param root_dir: path to directory where images stored
    :param output_dir: output directory
    """
    for img_name in tqdm(os.listdir(root_dir)):
        if img_name.endswith(('.jpeg', 'jpg', '.png')):
            img = cv2.imread(os.path.join(root_dir, img_name))
            img = iaa.fliplr(img)
            img_name = img_name.split('.')[0] + '_flipped.jpeg'
            cv2.imwrite(os.path.join(output_dir, img_name), img)


def resize_all_images(root_dir, output_dir, size=(224, 224)):
    """
    Resize all images from specified directory, copy and save them to output directory
    :param root_dir: directory where images stored
    :param output_dir: output directory
    :param size: ``Iterable(int, int)``, size transformation
    """
    if not (isinstance(size, collections.abc.Iterable) and all(isinstance(i, int) for i in size) and len(size) == 2):
        raise TypeError('the size must be Iterable(int, int)')

    aug_resize = iaa.Resize(size=size)

    for cls in os.listdir(root_dir):
        save_path = os.path.join(output_dir, cls)
        if not os.path.exists(save_path):
            os.mkdir(os.path.join(output_dir, cls))

        path_types = os.path.join(root_dir, cls)
        for img_name in tqdm(os.listdir(path_types)):
            if img_name.endswith(('.jpeg', 'jpg', '.png')):
                img = cv2.imread(os.path.join(path_types,  img_name))
                img = aug_resize.augment_image(img)
                cv2.imwrite(os.path.join(save_path, img_name), img)
