import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class XRayDataset(Dataset):
    """X-Ray dataset"""

    def __init__(self, df, transforms=True):
        """
        :param df: ``DataFrame`` with columns [img_path, label]
        :param transforms: if True then allow to transform
        """
        self.df = df
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = torch.tensor([0]) if self.df.iloc[idx, 1] == 'NORMAL' else torch.tensor([1])
        img = Image.open(img_path).convert('RGB')
        if transforms:
            img = self.img_to_tensor(img)

        return img, label

    def __len__(self):
        return len(self.df)

    @property
    def img_to_tensor(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                                               [0.229, 0.224, 0.225])])
