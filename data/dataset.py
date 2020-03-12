import kornia
import torch
import cv2
from torch.utils.data import Dataset


class XRayDataset(Dataset):
    """X-Ray dataset"""
    def __init__(self, df, transforms=True):
        """
        :param df: ``DataFrame`` with columns [img_path, label]
        :param transforms: if True then allow to transform
        """
        self.df = df

        if transforms:
            self.transform = self.aug_transform

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = torch.tensor([0]) if self.df.iloc[idx, 1] == 'NORMAL' else torch.tensor([1])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = kornia.image_to_tensor(img)
        return img, label

    def __len__(self):
        return len(self.df)

    @property
    def aug_transform(self):
        return torch.nn.Sequential(
            kornia.color.AdjustBrightness(0.5)
        )
