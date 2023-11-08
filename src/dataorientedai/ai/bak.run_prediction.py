# %%
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from dataorientedai.ai import SimpleCNN


def run_prediction(kwargs):
    device = kwargs.get("device", "cpu")
    mean = kwargs.get("mean", 0.5)
    std = kwargs.get("std", 0.5)
    path_model_state_dict = kwargs.get("path_model_state_dict", None)
    path_img_in = kwargs.get("path_img_in", "./00000.png")
    path_img_out = kwargs.get("path_img_out", "./00000_out.png")

    model = SimpleCNN(
        n_in=1,
        n_latent=128,
        n_out=11,
    ).to(device)
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
        plt.imshow(image_out, cmap="jet", vmin=0, vmax=10)
        plt.show()
        # image_out = postprocess(output)
        # plot_sample(image_out)
        # return image_out

    # image = np.random.randn(28, 28)
    # image_out = predict(model_in, image)
    # plot_image(image_out)
    plt.imshow(image_out)
    plt.show()
    plt.imsave(path_img_out, image_out)


if __name__ == "__main__":
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
