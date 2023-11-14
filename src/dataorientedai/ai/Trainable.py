# %%
import abc
import threading
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt
from PIL import Image

from dataorientedai.ai import MnistDatasetCNN, SimpleCNN, plot_sample
from dataorientedai.core.BlockingCollection import BlockingCollection
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IDictionary import IDictionary
from dataorientedai.core.interfaces.IUObject import IUObject
from dataorientedai.core.IoC import InitScopeBasedIoCImplementationCmd, IoC
from dataorientedai.core.UObject import UObject

# %%
# %%
# class FullModelSaveCmd(ICommand):
#     def __init__(self, o: IFullModelSavableAdapter):
#         self.o = o
#     def execute(self):
#         model = self.o.get_model('model')
#         global_step = self.o.get_global_step()
#         every_n_step = self.o.get_every_n_step()
#         if global_step % 100 == 0:
#             torch.save(model, "./model_full.pt")

# class ITrainProcessable(abc.ABC):
#     @abc.abstractmethod
#     def can_continue(self):
#         pass

#     @abc.abstractmethod
#     def process(self):
#         pass


# class TrainProcessable(ITrainProcessable):
#     def __init__(self, context: IDictionary):
#         self.context = context

#     def can_continue(self):
#         return self.context["can_continue"]

#     def process(self):
#         process = self.context["process"]
#         process()


# class TrainProcessor:
#     def __init__(self, context: ITrainProcessable):
#         self.train_processable = context
#         self.thread = threading.Thread(target=self.looping)
#         self.thread.start()

#     def wait(self):
#         # можем навечно заблокироваться в этом join;
#         # на такие случаи надо добавить time-outы
#         self.thread.join()

#     def looping(self):
#         while self.train_processable.can_continue():
#             self.train_processable.process()


# class InitTrainProcessorContextCmd(ICommand):
#     def __init__(self, context: IDictionary):
#         self.context = context

#     def execute(self):
#         can_continue = True
#         queue = BlockingCollection()

#         def process():
#             cmd = queue.get()
#             try:
#                 cmd.execute()
#                 # print(f"Executed: {cmd}")
#             except Exception as e:
#                 exc = type(e)
#                 try:
#                     # handler.handle(cmd, exc)
#                     # ExceptionHandler.handle(cmd, exc).execute()
#                     # IoC.resolve("ExceptionHandler", cmd, exc).execute()
#                     print(f"Error! {exc} in {type(cmd)}")
#                 except Exception as e:
#                     print(f"Fatal error! {exc} in {type(cmd)}")

#         self.context["can_continue"] = can_continue
#         self.context["queue"] = queue
#         self.context["process"] = process
#         self.context["thread_join_timeout"] = 2


# context = ContextDictionary()
# InitEventLoopContextCmd(context).execute()


# %%
class ITrainable(abc.ABC):
    def __init__(self, o: IUObject):
        self.o = o

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def set_model(self, model):
        pass

    @abc.abstractmethod
    def get_device(self):
        pass

    @abc.abstractmethod
    def get_epochs(self):
        pass

    @abc.abstractmethod
    def get_loss_fn(self):
        pass

    @abc.abstractmethod
    def get_optimizer(self):
        pass

    @abc.abstractmethod
    def get_train_dataloader(self):
        pass


class TrainableAdapter(ITrainable):
    def __init__(self, o: IUObject):
        self.o = o

    def get_model(self):
        return self.o.__getitem__("model")

    def set_model(self, model):
        return self.o.__setitem__("model", model)

    def get_device(self):
        return self.o.__getitem__("device")

    def get_epochs(self):
        return self.o.__getitem__("epochs")

    def get_loss_fn(self):
        return self.o.__getitem__("loss_fn")

    def get_optimizer(self):
        return self.o.__getitem__("optimizer")

    def get_train_dataloader(self):
        return self.o.__getitem__("train_dataloader")

    def get_root(self):
        return self.o.__getitem__("root")


class TrainCmd(ICommand):
    def __init__(self, o: TrainableAdapter):
        self.o = o

    def execute(self):
        device = self.o.get_device()
        model = self.o.get_model()
        epochs = self.o.get_epochs()
        loss_fn = self.o.get_loss_fn()
        optimizer = self.o.get_optimizer()
        train_dataloader = self.o.get_train_dataloader()
        root = self.o.get_root()

        global_step = 0
        global_history = defaultdict(list)
        stop = False
        epoch = 0
        while not stop:
            for image, mask in train_dataloader:
                image = image.to(device)
                mask = mask.to(device)
                pred = model(image)
                loss = loss_fn(pred, mask.long())
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                self.o.set_model(model)

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
                if global_step % 100 == 0:
                    torch.save(model, str(root / "models/model_full.pt"))
                    torch.save(
                        model.state_dict(), str(root / "models/model_state_dict.pth")
                    )
                global_step += 1

            epoch += 1
            if epoch == epochs:
                stop = True


class InitTrainableObjectCmd(ICommand):
    def __init__(self, o: IUObject):
        self.o = o

    def execute(self):
        self.o["root"] = Path(
            "/home/rinkorn/space/prog/python/free/project-dataorientedai/"
        )
        self.o["epochs"] = 2
        self.o["device"] = "cuda:0"
        self.o["batch_size"] = 128
        self.o["learning_rate"] = 3e-4
        self.o["mppxl_current"] = 1.0
        self.o["mppxl_target"] = 1.0
        self.o["mean"] = 0.13092535192648502
        self.o["std"] = 0.3084485240270358
        self.o["scale_factor"] = self.o["mppxl_current"] / self.o["mppxl_target"]
        self.o["transforms"] = A.Compose(
            [
                A.Normalize(
                    mean=self.o["mean"],
                    std=self.o["std"],
                    max_pixel_value=1.0,
                ),
                A.RandomScale(
                    scale_limit=[
                        self.o["scale_factor"] - 1.0,
                        self.o["scale_factor"] - 1.0,
                    ],
                    interpolation=cv2.INTER_AREA,
                    p=1,
                ),
                ToTensorV2(),
            ]
        )
        self.o["train_dataset"] = MnistDatasetCNN(
            path_x_train=str(self.o["root"] / "data/processed/mnist-images/x_train"),
            path_y_train=str(self.o["root"] / "data/processed/mnist-images/y_train"),
            transform=self.o["transforms"],
        )
        self.o["train_dataloader"] = torch.utils.data.DataLoader(
            dataset=self.o["train_dataset"],
            batch_size=self.o["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        self.o["model"] = SimpleCNN(
            n_in=1,
            n_latent=128,
            n_out=11,
        ).to(self.o["device"])
        self.o["optimizer"] = torch.optim.AdamW(
            self.o["model"].parameters(),
            lr=self.o["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False,
        )
        self.o["loss_fn"] = smp.losses.DiceLoss(
            mode="multiclass",
            classes=11,
            log_loss=False,
            from_logits=True,
            smooth=0.0,
            eps=1e-8,
            ignore_index=None,
        )


class RegisterTrainableObjectCmd(ICommand):
    def execute(self):
        obj = UObject()
        IoC.resolve(
            "IoC.register",
            "Objects.trainable_object_1",
            lambda *args: obj,
        ).execute()


# if __name__ == "__main__":
#     obj = UObject()
#     InitTrainableObject(obj).execute()
#     TrainCmd(TrainableAdapter(obj)).execute()

if __name__ == "__main__":
    InitScopeBasedIoCImplementationCmd().execute()
    scope = IoC.resolve("scopes.new", IoC.resolve("scopes.root"))
    IoC.resolve(
        "scopes.current.set",
        scope,
    ).execute()

    RegisterTrainableObjectCmd().execute()
    InitTrainableObjectCmd(IoC.resolve("Objects.trainable_object_1")).execute()
    TrainCmd(TrainableAdapter(IoC.resolve("Objects.trainable_object_1"))).execute()
