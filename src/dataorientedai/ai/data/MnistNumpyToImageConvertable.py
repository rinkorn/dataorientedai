# %%
import abc
from pathlib import Path

import cv2
import numpy as np
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IUObject import IUObject
from dataorientedai.core.UObject import UObject


class IMnistNumpyToImageConvertable(abc.ABC):
    @abc.abstractmethod
    def get_file_in(self):
        pass

    @abc.abstractmethod
    def get_path_out(self):
        pass


class MnistNumpyToImageConvertableAdapter(IMnistNumpyToImageConvertable):
    def __init__(self, o: IUObject):
        self.o = o

    def get_file_in(self):
        return self.o.__getitem__("file_in")

    def get_path_out(self):
        return self.o.__getitem__("path_out")


class MnistNumpyToImageConvertCmd(ICommand):
    def __init__(self, o: IMnistNumpyToImageConvertable):
        self.o = o

    def execute(self):
        file_in = Path(self.o.get_file_in())
        path_out = Path(self.o.get_path_out())

        if not file_in.exists():
            raise FileExistsError()
        if not path_out.exists():
            path_out.mkdir(parents=True, exist_ok=True)

        paths = {
            "x_train": path_out / "x_train",
            "x_test": path_out / "x_test",
            "y_train": path_out / "y_train",
            "y_test": path_out / "y_test",
        }
        for p in paths.values():
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)

        def loop_saving(paths, x_key, y_key):
            npz_file = np.load(file_in)
            X = npz_file[x_key]
            Y = npz_file[y_key]
            Y = np.where(Y == 0, 10, Y)
            for i in range(X.shape[0]):
                x_img = X[i, :, :]
                y_img = (x_img > 40.0) * Y[i]
                x_img = x_img[:, :, None]
                y_img = y_img[:, :, None]
                x_fn_out = str(paths[x_key] / f"{i:05d}.png")
                y_fn_out = str(paths[y_key] / f"{i:05d}.png")
                x_img = x_img.astype("uint8")
                y_img = y_img.astype("uint8")
                cv2.imwrite(x_fn_out, x_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(y_fn_out, y_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        loop_saving(paths, "x_train", "y_train")
        loop_saving(paths, "x_test", "y_test")


class InitMnistNumpyToImageConvertableObjectCmd(ICommand):
    def __init__(self, o: IUObject):
        self.o = o

    def execute(self):
        path = Path("/home/rinkorn/space/prog/python/free/project-dataorientedai")
        self.o.__setitem__("file_in", path / "data/processed/mnist-numpy/mnist.npz")
        self.o.__setitem__("path_out", path / "data/processed/mnist-images/")


if __name__ == "__main__":
    obj = UObject()
    InitMnistNumpyToImageConvertableObjectCmd(obj).execute()
    convertable_obj = MnistNumpyToImageConvertableAdapter(obj)
    cmd = MnistNumpyToImageConvertCmd(convertable_obj)
    cmd.execute()
