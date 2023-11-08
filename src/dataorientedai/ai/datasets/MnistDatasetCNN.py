# %%
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2
from dataorientedai.ai.common import plot_sample
from matplotlib import pyplot as plt


# %%
def rescale_sample(mppxl_current, mppxl_target, **kwargs):
    # https://robocraft.ru/computervision/3956
    scale_factor = mppxl_current / mppxl_target
    aug = A.RandomScale(
        scale_limit=[scale_factor - 1.0, scale_factor - 1.0],
        interpolation=cv2.INTER_AREA,
        p=1,
    )
    return aug


class MnistDatasetCNN(torch.utils.data.Dataset):
    """
    Custom MNIST dataset for instance segmentation
    """

    DFLT_MPPXL = 5.45
    DFLT_CLASSES = {
        "background": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "0": 10,
    }

    def __init__(
        self,
        path_x_train="./data/mnist/x_train",
        path_y_train="./data/mnist/y_train",
        transform=None,
        classes=DFLT_CLASSES,
    ):
        self.path_x_train = Path(path_x_train)
        self.path_y_train = Path(path_y_train)
        self.transform = transform
        self.classes = classes
        self.num_classes = len(self.classes)
        self.flist_x_train = list(self.path_x_train.glob("*.png"))

    def __getitem__(self, index):
        fn_x = self.flist_x_train[index]
        fn_y = self.path_y_train / fn_x.name
        image = plt.imread(str(fn_x), "png")
        mask = plt.imread(str(fn_y), "png")
        image = image * 255.0
        mask = mask * 255.0
        if self.transform is not None:
            data_dict = self.transform(image=image, mask=mask)
            image, mask = data_dict["image"], data_dict["mask"]
        return image, mask

    def __len__(self):
        return len(self.flist_x_train)


# %%
if __name__ == "__main__":
    # mppxl_current = 2.286
    mppxl_current = 1.0
    mppxl_target = 1.0
    scale_factor = mppxl_current / mppxl_target
    transforms = A.Compose(
        [
            A.Normalize(
                mean=0.13092535192648502,
                std=0.3084485240270358,
                max_pixel_value=255.0,
            ),
            A.RandomScale(
                scale_limit=[scale_factor - 1.0, scale_factor - 1.0],
                interpolation=cv2.INTER_AREA,
                p=1,
            ),
            ToTensorV2(),
        ]
    )
    train_dataset = MnistDatasetCNN(
        path_x_train="/workspaces/ml-mnist-segmentation/data/processed/mnist-images/x_train",
        path_y_train="/workspaces/ml-mnist-segmentation/data/processed/mnist-images/y_train",
        transform=transforms,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    for step, (image, mask) in enumerate(train_dataloader):
        # image = torch.unsqueeze(image, dim=1)
        mask = torch.unsqueeze(mask, dim=1)
        print(image.dtype)
        print(mask.dtype)
        print(image.shape)
        print(mask.shape)
        plot_sample(image, mask)
        print(image.shape)
        print(mask.shape)
        print(image.mean())
        print(image.std())
        break
