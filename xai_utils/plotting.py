"""Plotting helpers: rows of heatmaps next to the image they belong to.

Every heatmap is stretched between its own minimum and maximum, so brightness
compares positions within a panel, not across panels. Coarse grids are drawn
as blocks, one block per feature cell.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def _draw(ax, image, cmap='inferno'):
    """One image (PIL, array or tensor) or heatmap on one axis, without ticks."""
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    coarse = image.ndim == 2 and image.shape[0] < 64
    ax.imshow(image, cmap=cmap if image.ndim == 2 else None, interpolation='nearest' if coarse else 'antialiased')
    ax.set_xticks([])
    ax.set_yticks([])


def show_images(images, titles, size=2.6):
    """A single row of images or heatmaps with titles."""
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size), squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        _draw(ax, image)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    plt.show()


def show_rows(image, rows, columns, cmaps=None, size=2.2):
    """One row per model: the image, then one heatmap per column.

    image:   the image shown in the first column
    rows:    list of (row label, [heatmap per column])
    columns: column titles
    cmaps:   colormap per column (default 'inferno')
    """
    cmaps = cmaps or ['inferno'] * len(columns)
    fig, axes = plt.subplots(len(rows), len(columns) + 1, figsize=(size * (len(columns) + 1), size * len(rows)),
                             squeeze=False)
    for r, (label, maps) in enumerate(rows):
        _draw(axes[r, 0], image)
        axes[r, 0].set_ylabel(label, fontsize=9)
        for c, heatmap in enumerate(maps):
            _draw(axes[r, c + 1], heatmap, cmaps[c])
    for c, title in enumerate(['Image'] + list(columns)):
        axes[0, c].set_title(title, fontsize=10)
    fig.tight_layout()
    plt.show()
