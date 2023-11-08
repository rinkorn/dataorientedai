# %%
import shutil
from pathlib import Path

import click


def unpack_archive(fn_in: str, path_out: str):
    fn_in = Path(fn_in)
    path_out = Path(path_out)

    if not fn_in.exists():
        raise FileExistsError()

    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(fn_in, path_out)


@click.command()
@click.argument("fn_in", type=click.Path())
@click.argument("path_out", type=click.Path())
def main(fn_in: str, path_out: str):
    unpack_archive(fn_in, path_out)


if __name__ == "__main__":
    # main()
    # sys.argv = ['', '--fn_in', 'mnist.zip', '--fn_out', 'mnist-ubyte/']
    path = Path("/home/rinkorn/space/prog/python/free/project-dataorientedai/data")
    fn_in = path / "raw/mnist.zip"
    fn_out = path / "processed/mnist-ubyte/"
    unpack_archive(str(fn_in), str(fn_out))
