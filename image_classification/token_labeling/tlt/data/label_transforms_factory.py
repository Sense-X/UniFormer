""" Transforms Factory

Adapted for token labeling
"""
import math

import torch
from torchvision import transforms

from .random_augment_label import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from timm.data.random_erasing import RandomErasing
import random

import torchvision
from torchvision.transforms import functional as torchvision_F
from PIL import Image

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)

class ComposeWithLabel(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithLabel, self).__init__(**kwargs)

    def __call__(self, img, label_map):
        for t in self.transforms:
            if type(t).__name__ == 'RandomHorizontalFlipWithLabel':
                img, label_map = t(img, label_map)
            elif type(t).__name__ == 'RandomVerticalFlipWithLabel':
                img, label_map = t(img, label_map)
            elif type(t).__name__ == 'RandAugment':
                img, label_map = t(img, label_map)
            elif type(t).__name__ == 'RandomResizedCropAndInterpolationWithCoords':
                # should ensure RandomResizedCropWithCoords after all trabsformation
                img, label_map = t(img, label_map)
            else:
                img = t(img)
        return img, label_map

class RandomResizedCropAndInterpolationWithCoords(RandomResizedCropAndInterpolation):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, label_map):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        coords = (i / img.size[1],
                  j / img.size[0],
                  h / img.size[1],
                  w / img.size[0])
        coords_map = torch.zeros_like(label_map[0:1])
        # trick to store coords_map is label_map
        coords_map[0,0,0,0],coords_map[0,0,0,1],coords_map[0,0,0,2],coords_map[0,0,0,3] = coords
        label_map = torch.cat([label_map, coords_map])
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return torchvision_F.resized_crop(img, i, j, h, w, self.size,
                                 interpolation), label_map

class RandomHorizontalFlipWithLabel(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self, **kwargs):
        super(RandomHorizontalFlipWithLabel, self).__init__(**kwargs)

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            return torchvision_F.hflip(img), label.flip(3)
        return img, label

class RandomVerticalFlipWithLabel(torchvision.transforms.RandomVerticalFlip):
    def __init__(self, **kwargs):
        super(RandomVerticalFlipWithLabel, self).__init__(**kwargs)

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            return torchvision_F.vflip(img), label.flip(2)
        return img, label


def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range

    primary_tfl=[]
    if hflip > 0.:
        primary_tfl += [RandomHorizontalFlipWithLabel(p=hflip)]
    if vflip > 0.:
        primary_tfl += [RandomVerticalFlipWithLabel(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]

    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = [RandomResizedCropAndInterpolationWithCoords(size=img_size, scale=scale, ratio=ratio, interpolation=interpolation)]

    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))
    return ComposeWithLabel(transforms=primary_tfl + secondary_tfl + final_tfl)


def create_token_label_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False,):

    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    transform = transforms_imagenet_train(
        img_size,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        use_prefetcher=use_prefetcher,
        mean=mean,
        std=std,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=separate)

    return transform
