# %%
import abc
import shutil
from pathlib import Path

from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IUObject import IUObject
from dataorientedai.core.UObject import UObject


class IUnzippable(abc.ABC):
    @abc.abstractmethod
    def get_file_in(self):
        pass

    @abc.abstractmethod
    def get_path_out(self):
        pass


class UnzippableAdapter(IUnzippable):
    def __init__(self, o: IUObject):
        self.o = o

    def get_file_in(self):
        return self.o.__getitem__("file_in")

    def get_path_out(self):
        return self.o.__getitem__("path_out")


class UnzipCmd(ICommand):
    def __init__(self, o: IUnzippable):
        self.o = o

    def execute(self):
        file_in = Path(self.o.get_file_in())
        path_out = Path(self.o.get_path_out())

        if not file_in.exists():
            raise FileExistsError()

        if not path_out.exists():
            path_out.mkdir(parents=True, exist_ok=True)

        shutil.unpack_archive(file_in, path_out)


class InitUnzippableObject(ICommand):
    def __init__(self, o: IUObject):
        self.o = o

    def execute(self):
        project_path = Path(
            "/home/rinkorn/space/prog/python/free/project-dataorientedai"
        )
        self.o.__setitem__("file_in", project_path / "data/raw/mnist.zip")
        self.o.__setitem__("path_out", project_path / "data/processed/mnist-ubyte/")


if __name__ == "__main__":
    obj = UObject()
    InitUnzippableObject(obj).execute()
    UnzipCmd(UnzippableAdapter(obj)).execute()
