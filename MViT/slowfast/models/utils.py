# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from logging import NOTSET
import slowfast.utils.logging as logging
import numpy as np
import torch.nn.functional as F
from einops import rearrange

logger = logging.get_logger(__name__)


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        logger.info(f"min width {min_width}")
        logger.info(f"width {width} divisor {divisor}")
        logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def validate_checkpoint_wrapper_import(checkpoint_wrapper):
    """
    Check if checkpoint_wrapper is imported.
    """
    if checkpoint_wrapper is None:
        raise ImportError("Please install fairscale.")



def gen_visualization(image, anchor_size, num_anchors, stride, indices):
    # image: 1 c h w
    # indices: 1 1

    # 1 3x4x4 56 56
    image2 = rearrange(image, 'b c (nh ps1) (nw ps2) -> b (c ps1 ps2) (nh nw)', ps1=4, ps2=4)
    
    indices = F.one_hot(indices, num_classes=num_anchors).float()
    indices = indices.repeat(1, anchor_size*anchor_size, 1)
    # b 3x4x4 56 56
    indices = F.fold(indices, output_size=(56, 56), kernel_size=anchor_size, stride=stride)
    masked_image = image2 * indices + image2 * (1 - indices) * 0.2
    masked_image = rearrange(masked_image, 'b (c ps1 ps2) (nh nw) -> b c (nh ps1) (nw ps2)', ps1=4, ps2=4, nh=56)

    image = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
    masked_image = masked_image.detach().cpu().numpy()[0].transpose(1, 2, 0)

    viz = np.concatenate((image, masked_image), axis=1)

    return viz


def visualize_temporal(videos, indices):
    # videos: b t c h w
    # indices: b 1
    videos = videos[0:1, :]
    indices = indices[0:1, :]

    # b 1 t
    indices = F.one_hot(indices, num_classes=videos.size(1)).float()
    indices = indices.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, videos.size(2), videos.size(3), videos.size(4))
    masked_videos = videos * indices + videos * (1 - indices) * 0.2
    masked_videos = rearrange(masked_videos, 'b t c h w -> b h (t w) c')
    masked_videos = masked_videos.detach().cpu().numpy()

    return masked_videos[0]

     



def visualize_spatial(videos, indices):
    # videos: b t c h w
    # indices: b t 1
    videos = videos[0:1, :]
    indices = indices[0:1, :]

    masked_images = []
    for i in range(videos.size(1)):
        image = videos[:, i]
        indices_ = indices[:, i]
        masked_image = gen_visualization(image, 40, 9, 8, indices_)
        masked_images.append(masked_image)
    
    masked_images = np.concatenate(masked_images, axis=0)
    return masked_images

