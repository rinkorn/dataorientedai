# %%
import abc
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from dataorientedai.ai import SimpleCNN
from dataorientedai.core.Adapter import Adapter
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IUObject import IUObject
from dataorientedai.core.IoC import IoC
from dataorientedai.core.UObject import UObject


# %%
class IPredictable(abc.ABC):
    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def get_device(self):
        pass

    @abc.abstractmethod
    def get_path_img_in(self):
        pass

    @abc.abstractmethod
    def get_path_model_state_dict(self):
        pass

    @abc.abstractmethod
    def get_mean(self):
        pass

    @abc.abstractmethod
    def get_std(self):
        pass


class PredictableAdapter(IPredictable):
    def __init__(self, o: IUObject):
        self.o = o

    def get_model(self):
        return self.o.__getitem__("model")

    def get_device(self):
        return self.o.__getitem__("device")

    def get_path_img_in(self):
        return self.o.__getitem__("path_img_in")

    def get_path_model_state_dict(self):
        return self.o.__getitem__("path_model_state_dict")

    def get_mean(self):
        return self.o.__getitem__("mean")

    def get_std(self):
        return self.o.__getitem__("std")


class PredictCmd(ICommand):
    def __init__(self, o: PredictableAdapter):
        self.o = o

    def execute(self):
        device = self.o.get_device()
        path_img_in = self.o.get_path_img_in()
        path_model_state_dict = self.o.get_path_model_state_dict()
        mean = self.o.get_mean()
        std = self.o.get_std()
        model = self.o.get_model()

        state_dict = torch.load(path_model_state_dict)
        model.load_state_dict(state_dict)
        # model = LitModel(1, 128, 11)
        # model = torch.jit.load(path_in)
        # model = onnx.load(str(path_in))
        # # Check that the model is well formed
        # onnx.checker.check_model(model)
        # # Print a human readable representation of the graph
        # print(onnx.helper.printable_graph(model.graph))
        # # model = torch.load(str(path_in))

        image = Image.open(path_img_in)
        image = np.asarray(image)
        # image = plt.imread('image.png')
        # image = cv2.imread(fname)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # model.eval()
        model.train(False)
        # torch.set_float32_matmul_precision("medium")
        with torch.no_grad():
            image = (image - mean) / std
            image = torch.as_tensor(image).to(device)
            image = image.unsqueeze(0).unsqueeze(0)
            image_out = model(image.float())
            image_out = image_out[0, ...].permute(1, 2, 0).softmax(-1).argmax(-1)
            image_out = image_out.cpu().detach().numpy()
            # image_out = (image_out * std + mean) * 255
            image_out = image_out.astype("uint8")
            # plt.imshow(image_out, cmap="jet", vmin=0, vmax=10)
            # plt.colorbar()
            # plt.show()


class RegisterUObjectCmd(ICommand):
    def __init__(self, registration_name: str) -> None:
        self.registration_name = registration_name

    def execute(self) -> None:
        obj = IoC.resolve("UObject")
        IoC.resolve(
            "IoC.register",
            self.registration_name,
            lambda *args: obj,
        ).execute()


class InitPredictableObjectCmd(ICommand):
    def __init__(self, o: IUObject):
        self.o = o

    def execute(self):
        self.o["device"] = "cuda:0"
        self.o["root"] = Path(
            "/home/rinkorn/space/prog/python/free/project-dataorientedai/"
        )
        self.o["mean"] = 0.13092535192648502
        self.o["std"] = 0.3084485240270358
        self.o["path_model_state_dict"] = self.o["root"] / "models/model_state_dict.pth"
        self.o["path_img_in"] = self.o["root"] / "data/processed/mnist-images/x_train/"
        self.o["path_img_in"] = self.o["path_img_in"] / "00004.png"
        self.o["path_img_out"] = self.o["root"] / "00004_out.png"
        self.o["model"] = SimpleCNN(n_in=1, n_latent=128, n_out=11).to(self.o["device"])


# if __name__ == "__main__":
#     obj = UObject()
#     InitPredictableObject(obj).execute()
#     PredictCmd(PredictableAdapter(obj)).execute()

if __name__ == "__main__":
    from dataorientedai.core.IoC import InitScopeBasedIoCImplementationCmd

    # init IoC
    InitScopeBasedIoCImplementationCmd().execute()

    # init Scope
    IoC.resolve(
        "Scopes.current.set",
        IoC.resolve("Scopes.new", IoC.resolve("Scopes.root")),
    ).execute()

    # registration
    IoC.resolve(
        "IoC.register",
        "UObject",
        lambda *args: UObject(),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "Commands.RegisterUObjectCmd",
        lambda *args: RegisterUObjectCmd(*args),
    ).execute()

    IoC.resolve(
        "IoC.register",
        "Adapter",
        lambda *args: Adapter.generate(args[0])(args[1]),
    ).execute()

    IoC.resolve(
        "IoC.register",
        "Interfaces.IPredictable:device.get",
        lambda *args: args[0].__getitem__("device"),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "Interfaces.IPredictable:model.get",
        lambda *args: args[0].__getitem__("model"),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "Interfaces.IPredictable:path_img_in.get",
        lambda *args: args[0].__getitem__("path_img_in"),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "Interfaces.IPredictable:path_model_state_dict.get",
        lambda *args: args[0].__getitem__("path_model_state_dict"),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "Interfaces.IPredictable:mean.get",
        lambda *args: args[0].__getitem__("mean"),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "Interfaces.IPredictable:std.get",
        lambda *args: args[0].__getitem__("std"),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "Commands.InitPredictableObjectCmd",
        lambda *args: InitPredictableObjectCmd(*args),
    ).execute()

    IoC.resolve(
        "IoC.register",
        "Commands.PredictCmd",
        lambda *args: PredictCmd(*args),
    ).execute()

    # executing
    IoC.resolve(
        "Commands.RegisterUObjectCmd",
        "Objects.predictable",
    ).execute()
    IoC.resolve(
        "Commands.InitPredictableObjectCmd",
        IoC.resolve("Objects.predictable"),
    ).execute()

    # movable_obj = IoC.resolve("Adapter", IMovable, obj)
    IoC.resolve(
        "Commands.PredictCmd",
        IoC.resolve(
            "Adapter",
            IPredictable,
            IoC.resolve("Objects.predictable"),
        ),
    ).execute()

    IoC.resolve(
        "IoC.register",
        "Objects.Predict",
        lambda *args: PredictCmd(
            "Adapter",
            IPredictable,
            args[0],
        ),
    ).execute()

    # just printing
    scope = IoC.resolve("Scopes.current")
    print(id(scope), scope.dependencies._store.keys())

    scope = IoC.resolve("Scopes.root")
    print(id(scope), scope.dependencies._store.keys())
