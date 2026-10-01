# Deep Unboxing

Companion notebooks for my [blog](https://karay.me) posts on what deep vision
models see and how to explain it. Each notebook reproduces the figures and
numbers of one post.

| Notebook | Blog post |
|---|---|
| `01_label_free_maps.ipynb` | [What Does a Vision Model See Without Labels?](https://karay.me/2026/10/01/unsupervised-feature-attribution.html) |

## Setup

The project uses [uv](https://docs.astral.sh/uv/). From this folder:

```bash
uv sync
uv run jupyter lab
```

Then open `01_label_free_maps.ipynb` and run the cells from top to bottom.

A GPU helps but is not required. On first use, the notebook downloads 17
pretrained models (about 6 GB in total), most of them from the Hugging Face
Hub, and the AnyUp upsampler from GitHub. Each section only loads the models
it needs. To try things quickly, shorten the
model lists at the top of a section.

## What is in here

| Path | Contents |
|---|---|
| `images/` | the example images, at their original resolution; `edited/` holds the photo with the fish removed |
| `xai_utils/models.py` | the models and a wrapper that returns their feature grid |
| `xai_utils/lrp.py` | layer-wise relevance propagation that starts from the features |
| `xai_utils/plotting.py` | small plotting helpers |

Everything the posts explain, such as the label-free maps, RISE, RELAX, CAM and
Grad-CAM, is written out in the notebooks themselves, and so is loading the
images. `xai_utils/` only holds the plumbing around it.

The LRP rules in `xai_utils/lrp.py` are one reasonable set of choices for these
architectures. They are condensed from a larger research code base and
reproduce its relevance maps exactly for all models except DINOv2, which is
loaded here through timm instead of Meta's own code.

## What is not in here

The quantitative evaluation of the LaFAM paper (ImageNet-S and PASCAL VOC with
the Quantus metrics) is not reproduced here. See the
[LaFAM repository](https://github.com/karray/LaFAM) for that.
