# %%
from matplotlib import pyplot as plt


# %%
def plot_sample(image, mask, is_show=True, **kwargs):
    image = image[0, ...].permute(1, 2, 0).cpu().detach().numpy()
    mask = mask[0, ...].permute(1, 2, 0).cpu().detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(6, 2))
    ax[0].imshow(image, interpolation="None", **kwargs)
    ax[1].imshow(mask, interpolation="None", **kwargs)
    if is_show:
        plt.show()
    return fig, ax
