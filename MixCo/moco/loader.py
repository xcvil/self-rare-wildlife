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

class MixupThreeCropsTransform:
    """Take three random crops of one image as the query, key and extra query."""

    def __init__(self, base_transform, q_transform, k_transform, q2_transform, p, random_mixup=False):
        """p: probability to apply the mixup augmentation"""
        self.q_transform = q_transform
        self.k_transform = k_transform
        self.q2_transform = q2_transform
        self.base_transform = base_transform
        self.p = p
        self.random_mixup = random_mixup

    def __call__(self, x):
        mixup = random.uniform(0, 1)
        if mixup < self.p:
            x = self.base_transform(x)
            k = self.k_transform(x)
            q = self.q_transform(x)
            eq = self.q2_transform(x)
            if self.random_mixup:
                lam = random.uniform(0, 1)
                q = lam*q + (1-lam)*k
                eq = lam*q + (1-lam)*eq
            else:
                q = 0.5*q + 0.5*k
                eq = 0.5*q + 0.5*eq
        else:
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
