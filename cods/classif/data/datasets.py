"""Datasets for conformal classification tasks."""

import json
import os
import urllib.request
from typing import Callable, Dict

import torch
import torchvision.transforms as T
from timm.data.dataset import ImageDataset


class ClassificationDataset(ImageDataset):
    """Dataset for classification tasks with class index mapping and optional transforms."""

    def __init__(
        self,
        path: str,
        transforms: Callable = None,
        idx_to_cls: Dict[int, str] = None,
        **kwargs,
    ):
        """Initialize the ClassificationDataset.

        Args:
        ----
            path (str): Path to the dataset.
            transforms (Callable, optional): Transformations to apply to images.
            idx_to_cls (dict, optional): Mapping from class indices to class names.
            **kwargs: Additional arguments for the base dataset.

        Raises:
        ------
            ValueError: If idx_to_cls is not provided.

        """
        super().__init__(path, **kwargs)
        self._path = path
        self.transforms = transforms
        self.idx_to_cls = idx_to_cls
        if idx_to_cls is None:
            raise ValueError("idx_to_cls must be provided")

    def random_split(self, lengths, seed=0):
        """Randomly split the dataset into subsets of given lengths.

        Args:
        ----
            lengths (list): Lengths of splits.
            seed (int, optional): Random seed. Defaults to 0.

        Yields:
        ------
            ClassificationDataset: Subsets of the dataset.

        Raises:
        ------
            ValueError: If idx_to_cls is not set in the split.

        """
        generator = torch.Generator().manual_seed(seed)
        datasets = torch.utils.data.random_split(
            self,
            lengths=lengths,
            generator=generator,
        )
        for dataset in datasets:
            if not isinstance(dataset, ClassificationDataset) or dataset.idx_to_cls is None:
                raise ValueError("idx_to_cls should've been set!")
            dataset.idx_to_cls = self.idx_to_cls
            yield dataset

    def __getitem__(self, item):
        """Get an item from the dataset.

        Args:
        ----
            item (int): Index of the item.

        Returns:
        -------
            tuple: (image, label)

        """
        img, label = super().__getitem__(item)
        return img, label


class ImageNetDataset(ClassificationDataset):
    """Dataset for ImageNet with automatic class index mapping and default transforms."""

    def __init__(self, path: str, transforms: Callable = None, **kwargs):
        """Initialize the ImageNetDataset.

        Args:
        ----
            path (str): Path to the dataset.
            transforms (Callable, optional): Transformations to apply to images.
            **kwargs: Additional arguments for the base dataset.

        """
        tmp = json.loads(
            urllib.request.urlopen(
                "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
            ).read(),
        )
        wdnids = {int(k): v[0] for k, v in tmp.items()}
        self.wdnids = wdnids
        idx_to_cls = {int(k): v[1] for k, v in tmp.items()}
        if transforms is None:
            transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ],
            )
        super().__init__(
            path=path,
            transforms=transforms,
            idx_to_cls=idx_to_cls,
            **kwargs,
        )

    def __getitem__(self, item):
        """Get an item from the ImageNet dataset.

        Args:
        ----
            item (int): Index of the item.

        Returns:
        -------
            tuple: (image path, image, label)

        """
        img, label = super(ImageNetDataset, self).__getitem__(item)
        if self.transforms is not None:
            img = self.transforms(img)

        return os.path.join(self._path, str(self.filename(item))), img, label
