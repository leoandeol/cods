import json
import os
import urllib.request
from typing import Callable, Dict, Union

import torch
import torchvision.transforms as T
from timm.data.dataset import ImageDataset

# from PIL import Image
# from torch.utils.data import Dataset

# from torchvision.datasets import ImageNet


class ClassificationDataset(ImageDataset):
    def __init__(
        self,
        path: str,
        transforms: Union[Callable, None] = None,
        idx_to_cls: Union[Dict[int, str], None] = None,
        **kwargs
    ):
        super().__init__(path, **kwargs)
        self._path = path
        self.transforms = transforms
        self.idx_to_cls = idx_to_cls
        if idx_to_cls is None:
            raise ValueError("idx_to_cls must be provided")

    def random_split(self, lengths, seed=0):
        generator = torch.Generator().manual_seed(seed)
        datasets = torch.utils.data.random_split(
            self, lengths=lengths, generator=generator
        )
        for dataset in datasets:
            dataset.idx_to_cls = self.idx_to_cls
            yield dataset

    # do not alter
    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        return img, label


class ImageNetDataset(ClassificationDataset):
    def __init__(self, path: str, transforms: Union[Callable, None] = None, **kwargs):
        tmp = json.loads(
            urllib.request.urlopen(
                "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
            ).read()
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
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        super().__init__(
            path=path, transforms=transforms, idx_to_cls=idx_to_cls, **kwargs
        )

    def __getitem__(self, item):
        img, label = super(ImageNetDataset, self).__getitem__(item)
        if self.transforms is not None:
            img = self.transforms(img)

        return os.path.join(self._path, str(self.filename(item))), img, label
