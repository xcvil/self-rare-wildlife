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


class ThreeCropsTransform:
    """Take three random crops of one image as the query, key and extra query."""

    def __init__(self, base_transform, q_transform, k_transform, q2_transform):
        self.q_transform = q_transform
        self.k_transform = k_transform
        self.q2_transform = q2_transform
        self.base_transform = base_transform

    def __call__(self, x):
        x = self.base_transform(x)
        k = self.k_transform(x)
        q = self.q_transform(x)
        eq = self.q2_transform(x)
        return [q, k, eq]


class LooCCropsTransform:
    """Take three random crops of one image as the query, key and extra query."""

    def __init__(self, base_transform, q_transform, k_transform, q2_transform):
        self.q_transform = q_transform
        self.k_transform = k_transform
        self.q2_transform = q2_transform
        self.base_transform = base_transform

    def __call__(self, x):
        x = self.base_transform(x)
        k = self.k_transform(x)
        q = self.q_transform(x)
        eq = self.q2_transform(x)
        return [q, k, eq]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
