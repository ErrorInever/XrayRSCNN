import torch
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2


class XrayImageFolder(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]

        target = 0 if 'NORMAL' in path else 1

        return sample, target


class XRayDataset(Dataset):
    """X-Ray dataset"""

    def __init__(self, df, transforms=None):
        """
        :param df: ``DataFrame`` with columns [img_path, label]
        :param transforms: if True then allow to transform
        """
        self.df = df
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = torch.tensor([0]) if self.df.iloc[idx, 1] == 'NORMAL' else torch.tensor([1])
        img = cv2.imread(img_path)

        if self.transforms:
            img = self.transform(img)
        else:
            val_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            img = val_transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

    @property
    def transform(self):
        return transforms.Compose([
            iaa.Sequential([
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 4.0))),
                iaa.Fliplr(0.2),
                iaa.Affine(rotate=(-20, 20), mode='symmetric'),
                iaa.Multiply(0.50),
                iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                               iaa.CoarseDropout(0.1, size_percent=0.5)
                                               ])),
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
                iaa.Sometimes(0.10, iaa.SaltAndPepper(0.03), iaa.LogContrast(gain=0.5))
            ]).augment_image,
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.1581, 0.1562, 0.1562], std=[0.0756, 0.0751, 0.0751])
            #transforms.Normalize(mean=[0.4819, 0.4819, 0.4819], std=[0.2396, 0.2396, 0.2396])
        ])
