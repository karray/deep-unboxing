"""Plotting helper: a row of images or heatmaps with optional titles and label.

Every heatmap is stretched between its own minimum and maximum, so brightness
compares positions within a panel, not across panels.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(images, titles=None, ylabel='', cmap='inferno', size=2.4):
    """A single row of images or heatmaps with optional column titles and row label."""
    images = list(images)
    cmaps = [cmap] * len(images) if isinstance(cmap, str) else [None] * (len(images) - len(cmap)) + list(cmap)
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size), squeeze=False)
    axes[0, 0].set_ylabel(ylabel, fontsize=9)
    for ax, image, cm in zip(axes[0], images, cmaps):
        arr = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        ax.imshow(arr, cmap=cm if arr.ndim == 2 else None, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, title in zip(axes[0], titles or []):
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    plt.show()
