# %%
from pathlib import Path

import albumentations as A
import click
import cv2
import lightning as L
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch.transforms import ToTensorV2
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from torch.utils.data import DataLoader, random_split

from dataorientedai.ai.datasets.MnistDatasetCNN import MnistDatasetCNN
from dataorientedai.ai.models.SimpleCNN import SimpleCNN


# %%
class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_data: str,
        batch_size: int = 128,
        num_workers: int = 1,
    ):
        super().__init__()
        self.path_data = Path(path_data)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # some downloading here
        # all what to need prepare for learning[sync/async]
        pass

    def setup(self, stage=None):
        mppxl_current = 1.0
        mppxl_target = 1.0
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
        dataset = MnistDatasetCNN(
            path_x_train=self.path_data / "x_train",
            path_y_train=self.path_data / "y_train",
            transform=transforms,
        )
        self.train_ds, self.valid_ds, self.test_ds = random_split(
            dataset,
            [50000, 5000, 5000],
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
        )


# %%
class LitModel(pl.LightningModule):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.net = SimpleCNN(n_in=1, n_latent=128, n_out=11)
        self.loss_fn = loss_fn

    def forward(self, x):
        x = self.net(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False,
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y_real = train_batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y_real.long())
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_real = val_batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y_real.long())
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        x, y_real = test_batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y_real.long())
        self.log("test_loss", loss, sync_dist=True)


# %%
def run_training(kwargs):
    # if __name__ == "__main__":
    PATH_PROJECT = kwargs["path_project"]
    PATH_DATASET = PATH_PROJECT + "data/processed/mnist-images/"
    PATH_WEIGHTS = PATH_PROJECT + "models/weights/"
    PATH_PROJECT + "logs/tensorboard/"
    PATH_PROJECT + "logs/mlflow/"
    SEED_NUMBER = kwargs["seed_number"]

    # Initialize deterministic
    pl.seed_everything(SEED_NUMBER)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Initialize a dataloader
    data_loader = LitDataModule(
        path_data=PATH_DATASET,
        num_workers=4,
        batch_size=kwargs["batch_size"],
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
    # loss_fn = nn.CrossEntropyLoss()

    # Initialize a pl_model
    pl_model = LitModel(loss_fn)

    # Initialize callbacks
    save_every_n_train_steps = 50
    # tqdm_progress_bar = TQDMProgressBar(
    #     refresh_rate=1,
    # )
    rich_progress_theme = RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    )
    rich_progress_bar = RichProgressBar(
        theme=rich_progress_theme,
    )
    # lg_mlflow = MLFlowLogger(
    #     save_dir=PATH_MLFLOW,
    #     experiment_name="lightning_logs",
    #     tracking_uri="file:./logs/mlflow/",
    # )
    # cb_checkpoint1 = ModelCheckpoint(
    #     dirpath=PATH_MODELS,
    #     filename="model_{epoch}-{train_loss:.5f}",
    #     save_top_k=3,
    #     every_n_train_steps=save_every_n_train_steps,
    #     save_weights_only=False,
    #     mode="min",
    #     monitor="train_loss",
    #     # monitor="train_loss",
    # )
    cb_checkpoint2 = ModelCheckpoint(
        dirpath=PATH_WEIGHTS,
        filename="weights_{epoch}-{train_loss:.5f}",
        save_top_k=3,
        every_n_train_steps=save_every_n_train_steps,
        save_weights_only=True,
        mode="min",
        monitor="train_loss",
    )
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=5000,
    #     patience=40,
    #     verbose=False,
    #     mode="min",
    # )
    loggers = [
        # lg_mlflow,
    ]
    callbacks = [
        rich_progress_bar,
        # cb_early_stop,
        # cb_checkpoint1,
        cb_checkpoint2,
    ]

    # strategy = DDPStrategy(find_unused_parameters=True)
    # strategy = "ddp_notebook"
    # strategy = "ddp"

    # Restore only weights
    # RESTORE_CHECKPOINT = PATH_WEIGHTS + "weights_epoch=1333-train_loss=0.84433.ckpt"
    # pl_model.load_from_checkpoint(RESTORE_CHECKPOINT)
    # resume_from_checkpoint=PATH_FULL_RESTORING,

    # Initialize a trainer
    trainer = L.Trainer(
        # fast_dev_run=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        accelerator="auto",
        strategy="auto",
        devices="auto",
        num_nodes=1,
        # track_grad_norm=2,  # track the 2-norm
        precision="bf16-mixed",
        max_epochs=kwargs["epochs"],
        # limit_train_batches=10,
        limit_val_batches=10,
        # limit_test_batches=0.1,
        # val_check_interval=0.25,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        # strategy=strategy,
        # sync_batchnorm=True,
        log_every_n_steps=1,  # Control Logging Frequency
        logger=loggers,
        callbacks=callbacks,
        # accumulate_grad_batches={0: 4},
        # accumulate_grad_batches={1: 2, 4: 4, 10: 10, 40: 20, 100: 40, 200: 80},
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="value",
        # detect_anomaly=True,  # Detect autograd anomalies
        # amp_backend="native", # "apex"
        # auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
        # auto_lr_find='lr',  # special name for variable
        # num_nodes=1,
        # profiler="advanced",
        # reload_dataloaders_every_n_epochs = 10,
        # inference_mode=True
    )

    # Train the model
    trainer.fit(
        pl_model,
        data_loader,
        # ckpt_path=PATH_FULL_RESTORING,
    )

    # save the last checkpoint
    trainer.save_checkpoint(
        Path(PATH_PROJECT) / kwargs["path_models"] / "pl_model.ckpt"
    )
    # pl_model = LitModel.load_from_checkpoint(checkpoint_path="example.ckpt")

    # save the pl_model state
    torch.save(
        pl_model.state_dict(),
        Path(PATH_PROJECT) / kwargs["path_models"] / "pl_model.pt",
    )

    # save the net state
    torch.save(
        pl_model.net.state_dict(),
        Path(PATH_PROJECT) / kwargs["path_models"] / "pl_model.net.pt",
    )

    # save the onnx model
    pl_model.to_onnx(
        Path(PATH_PROJECT) / kwargs["path_models"] / "pl_model.onnx",
        torch.randn(1, 1, 28, 28),
        export_params=True,
    )

    # pl_model.to_torchscript(file_path="models/model.pt")
    pl_model.to_torchscript(
        Path(PATH_PROJECT) / kwargs["path_models"] / "pl_model.torchscripted.pt",
        method="trace",
        example_inputs=torch.randn(1, 1, 28, 28),
    )

    # extra_files = {'foo.txt': b'bar'}
    # torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)

    # torch.jit.save(
    #     pl_model.to_torchscript(
    #         file_path="models/torchscripted/model_trace.pt",
    #         method="trace",
    #         example_inputs=torch.randn(1, 1, 28, 28),
    #     )
    # )


# %%
@click.command()
@click.option("--seed_number", type=click.INT, default=42)
@click.option("--batch_size", type=click.INT, default=128)
@click.option("--epochs", type=click.INT, default=50)
@click.option("--path_models", type=click.Path(), default="models/")
def main(seed_number, batch_size, epochs, path_models):
    print("run_training")
    kwargs = {
        "seed_number": seed_number,
        "batch_size": batch_size,
        "epochs": epochs,
        "path_models": path_models,
    }
    run_training(kwargs)


if __name__ == "__main__":
    main()
