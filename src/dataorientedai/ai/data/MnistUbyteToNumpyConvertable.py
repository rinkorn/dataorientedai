# %%
import abc
from pathlib import Path

import idx2numpy
import numpy as np
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IUObject import IUObject
from dataorientedai.core.UObject import UObject


class IMnistUbyteToNumpyConvertable(abc.ABC):
    @abc.abstractmethod
    def get_path_in(self):
        pass

    @abc.abstractmethod
    def get_file_out(self):
        pass


class MnistUbyteToNumpyConvertableAdapter(abc.ABC):
    def __init__(self, o: IUObject):
        self.o = o

    def get_path_in(self):
        return self.o.__getitem__["path_in"]

    def get_file_out(self):
        return self.o.__getitem__["file_out"]


class MnistUbyteToNumpyConvertCmd(ICommand):
    def __init__(self, o: IMnistUbyteToNumpyConvertable):
        self.o = o

    def execute(self):
        path_in = Path(self.o.get_path_in())
        file_out = Path(self.o.get_file_out())

        if not path_in.exists():
            raise NotADirectoryError()

        if not file_out.parent.exists():
            file_out.parent.mkdir(parents=True, exist_ok=True)

        fn_x_train = path_in / "train-images-idx3-ubyte/train-images-idx3-ubyte"
        fn_y_train = path_in / "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
        fn_x_test = path_in / "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
        fn_y_test = path_in / "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

        x_train = idx2numpy.convert_from_file(str(fn_x_train))
        y_train = idx2numpy.convert_from_file(str(fn_y_train))
        x_test = idx2numpy.convert_from_file(str(fn_x_test))
        y_test = idx2numpy.convert_from_file(str(fn_y_test))

        np.savez(
            file_out,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )


if __name__ == "__main__":
    path = Path("/home/rinkorn/space/prog/python/free/project-dataorientedai")

    uobj = UObject()
    uobj.__setitem__("path_in", path / "data/processed/mnist-ubyte/")
    uobj.__setitem__("file_out", path / "data/processed/mnist-numpy/mnist.npz")

    convertable_obj = MnistUbyteToNumpyConvertableAdapter(uobj)
    MnistUbyteToNumpyConvertCmd(convertable_obj).execute()
