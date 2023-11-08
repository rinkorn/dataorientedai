# %%
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch.transforms import ToTensorV2
from dataorientedai.ai import MnistDatasetCNN, SimpleCNN, plot_sample


# %%
def main(kwargs):
    epochs = kwargs.get("epochs", 100)
    device = kwargs.get("device", "cuda:0")
    path_project = kwargs.get("path_project", "./")
    batch_size = kwargs.get("batch_size", 128)
    learning_rate = kwargs.get("learning_rate", 3e-4)
    mppxl_current = kwargs.get("mppxl_current", 1.0)
    mppxl_target = kwargs.get("mppxl_target", 1.0)

    scale_factor = mppxl_current / mppxl_target
    transforms = A.Compose(
        [
            A.Normalize(
                mean=0.13092535192648502,
                std=0.3084485240270358,
                max_pixel_value=1.0,
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
        path_x_train=str(path_project / "data/processed/mnist-images/x_train"),
        path_y_train=str(path_project / "data/processed/mnist-images/y_train"),
        transform=transforms,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    model = SimpleCNN(
        n_in=1,
        n_latent=128,
        n_out=11,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
    )
    loss_fn = smp.losses.DiceLoss(
        mode="multiclass",
        classes=11,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        eps=1e-8,
        ignore_index=None,
    )

    global_step = 0
    global_history = defaultdict(list)
    for epoch in range(epochs):
        for image, mask in train_dataloader:
            image = image.to(device)
            mask = mask.to(device)
            pred = model(image)
            loss = loss_fn(pred, mask.long())
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            loss_value = loss.cpu().detach().numpy()
            global_history["loss"].append(loss_value)
            if global_step % 100 == 0:
                print(
                    f"Epoch: {epoch}, "
                    f"global_step: {global_step:06d}, "
                    f"train_loss: {loss_value:.6f}"
                )
            if global_step % 100 == 0:
                img = image[0, ...].unsqueeze(0)
                msk = mask[0, ...].unsqueeze(0).unsqueeze(0)
                mask_argmax = pred[0, ...].argmax(0).unsqueeze(0).unsqueeze(0)
                fig, ax = plot_sample(
                    msk,
                    mask_argmax,
                    clim=(0.0, 10.0),
                    cmap="jet",
                )
            global_step += 1


# %%
if __name__ == "__main__":
    kwargs = {}
    kwargs["epochs"] = 100
    kwargs["device"] = "cuda:0"
    kwargs["path_project"] = Path(
        "/home/rinkorn/space/prog/python/free/project-dataorientedai/"
    )
    main(kwargs)
