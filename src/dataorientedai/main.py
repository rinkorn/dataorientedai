from pathlib import Path

import dataorientedai.ai._predict_model as pm
import dataorientedai.ai._train_litmodel as tm
from dataorientedai.ai.data.MnistNumpyToImageConvertable import (
    MnistNumpyToImageConvertableAdapter,
    MnistNumpyToImageConvertCmd,
)
from dataorientedai.ai.data.MnistUbyteToNumpyConvertable import (
    MnistUbyteToNumpyConvertableAdapter,
    MnistUbyteToNumpyConvertCmd,
)
from dataorientedai.ai.data.Unzippable import (
    InitUnzippableObject,
    UnzipCmd,
    UnzippableAdapter,
)
from dataorientedai.core.UObject import UObject

if __name__ == "__main__":
    root = Path("/home/rinkorn/space/prog/python/free/project-dataorientedai/")

    obj = UObject()
    InitUnzippableObject(obj).execute()
    UnzipCmd(UnzippableAdapter(obj)).execute()

    uobj = UObject()
    uobj.set_property("path_in", root / "data/processed/mnist-ubyte/")
    uobj.set_property("file_out", root / "data/processed/mnist-numpy/mnist.npz")
    convertable_obj = MnistUbyteToNumpyConvertableAdapter(uobj)
    MnistUbyteToNumpyConvertCmd(convertable_obj).execute()

    uobj = UObject()
    uobj.set_property("file_in", root / "data/processed/mnist-numpy/mnist.npz")
    uobj.set_property("path_out", root / "data/processed/mnist-images/")
    convertable_obj = MnistNumpyToImageConvertableAdapter(uobj)
    MnistNumpyToImageConvertCmd(convertable_obj).execute()


if __name__ == "__main__":
    from dataorientedai.ai.run_training import run_training

    kwargs = {}
    kwargs["epochs"] = 100
    kwargs["device"] = "cuda:0"
    kwargs["root"] = Path(
        "/home/rinkorn/space/prog/python/free/project-dataorientedai/"
    )
    kwargs["mean"] = 0.13092535192648502
    kwargs["std"] = 0.3084485240270358
    run_training(kwargs)

if __name__ == "__main__":
    from dataorientedai.ai.run_prediction import run_prediction

    kwargs = {}
    kwargs["device"] = "cuda:0"
    kwargs["root"] = Path(
        "/home/rinkorn/space/prog/python/free/project-dataorientedai/"
    )
    kwargs["mean"] = 0.13092535192648502
    kwargs["std"] = 0.3084485240270358
    kwargs["path_model_state_dict"] = kwargs["root"] / "models/model_state_dict.pth"
    kwargs["path_img_in"] = kwargs["root"] / "data/processed/mnist-images/x_train/"
    kwargs["path_img_in"] = kwargs["path_img_in"] / "00000.png"
    kwargs["path_img_out"] = kwargs["root"] / "00000_out.png"
    run_prediction(kwargs)
