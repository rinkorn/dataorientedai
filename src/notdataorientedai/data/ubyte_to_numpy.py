# %%
from pathlib import Path

import click
import idx2numpy
import numpy as np


def ubyte_to_numpy(path_in: str, path_out: str):
    path_in = Path(path_in)
    path_out = Path(path_out)

    if not path_in.exists():
        raise NotADirectoryError()

    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)

    fn_x_train = path_in / "train-images-idx3-ubyte/train-images-idx3-ubyte"
    fn_y_train = path_in / "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    fn_x_test = path_in / "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    fn_y_test = path_in / "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

    x_train = idx2numpy.convert_from_file(str(fn_x_train))
    y_train = idx2numpy.convert_from_file(str(fn_y_train))
    x_test = idx2numpy.convert_from_file(str(fn_x_test))
    y_test = idx2numpy.convert_from_file(str(fn_y_test))

    np.savez(
        path_out / "mnist.npz",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )


@click.command()
@click.argument("path_in", type=click.Path())
@click.argument("path_out", type=click.Path())
def main(path_in: str, path_out: str):
    ubyte_to_numpy(path_in, path_out)


if __name__ == "__main__":
    # main()
    path = Path("/home/rinkorn/space/prog/python/free/project-dataorientedai/data")
    ubyte_to_numpy(
        path_in=str(path / "processed/mnist-ubyte/"),
        path_out=str(path / "processed/mnist-numpy/"),
    )
