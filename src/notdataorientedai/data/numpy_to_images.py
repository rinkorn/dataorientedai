# %%
from pathlib import Path

import click
import cv2
import numpy as np


def numpy_to_images(path_in: str, path_out: str):
    path_in = Path(path_in)
    path_out = Path(path_out)
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
        npz_file = np.load(path_in / "mnist.npz")
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


@click.command()
@click.argument("path_in", type=click.Path())
@click.argument("path_out", type=click.Path())
def main(path_in, path_out):
    path_in = Path(path_in)
    path_out = Path(path_out)
    if not path_in.exists():
        raise NotADirectoryError()
    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)
    numpy_to_images(path_in, path_out)


if __name__ == "__main__":
    main()
    # path_data = Path("/workspaces/ml-mnist-segmentation/data")
    # path_in = path_data / "processed/mnist-numpy/"
    # path_out = path_data / "processed/mnist-images/"
    # numpy_to_images(path_in, path_out)
