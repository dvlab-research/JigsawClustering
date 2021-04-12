# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class JigCluTransform:
    def __init__(self, transform, cross = 0.0):
        self.transform = transform
        self.c = cross

    def __call__(self, x):
        h,w = x.size
        ch = self.c * h
        cw = self.c * w
        return [self.transform(x.crop((0,           0,          h//2+ch,    w//2+cw))),
                self.transform(x.crop((0,           w//2-cw,    h//2+ch,    w))),
                self.transform(x.crop((h//2-ch,     0,          h,          w//2+cw))),
                self.transform(x.crop((h//2-ch,     w//2-cw,    h,          w)))]

