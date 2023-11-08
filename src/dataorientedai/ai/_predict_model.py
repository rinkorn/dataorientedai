# %%
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from dataorientedai.ai._train_litmodel import LitModel
from dataorientedai.ai.common.load_image import load_image


def load_model(path_in):
    # model = LitModel(1, 128, 11)
    model = LitModel()
    state_dict = torch.load(path_in)
    model.load_state_dict(state_dict)
    model.eval()
    # model = torch.jit.load(path_in)
    # model = onnx.load(str(path_in))
    # # Check that the model is well formed
    # onnx.checker.check_model(model)
    # # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))
    # # model = torch.load(str(path_in))
    return model


def predict(model_in, image):
    mean = 0.13092535192648502
    std = 0.3084485240270358
    # torch.set_float32_matmul_precision("medium")
    model = load_model(model_in)
    model.train(False)
    with torch.no_grad():
        image = (image - mean) / std
        image = torch.as_tensor(image)
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
    return image_out


def main_func(model_in, image_in, image_out):
    # model = load_model(model_in)
    image = load_image(image_in)
    # image = np.random.randn(28, 28)
    image_out = predict(model_in, image)
    # plot_image(image_out)
    plt.imshow(image_out)
    plt.show()
    plt.imsave("image_out.png", image_out)
    # save_image(image_out, "image_out.png")


# @click.command()
# @click.option("--model_in", click.Path())
# @click.option("--image_in", click.Path())
# @click.option("--image_out", click.Path())
# def main(model_in, image_in, image_out):
#     main_func(model_in, image_in, image_out)


if __name__ == "__main__":
    # main()
    PATH_PROJECT = Path("/workspaces/ml-mnist-segmentation/")
    model_path = PATH_PROJECT / "models/checkpoints/SimpleNet.pt"
    # model_path = PATH_PROJECT / "models/onnx/export.onnx"
    # model_path = PATH_PROJECT / "models/torchscripted/model_traced.pt"
    image_in = PATH_PROJECT / "data/processed/mnist-images/x_train/00000.png"
    image_out = PATH_PROJECT / "data" / image_in.name
    main_func(str(model_path), str(image_in), str(image_out))
