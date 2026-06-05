import random
from logging import getLogger
from typing import Any, Callable, ClassVar
from xml.etree.ElementTree import parse as ET_parse

from PIL import Image
from torchvision.datasets import VOCDetection

logger = getLogger("cods")


class VOCDataset(VOCDetection):
    VOC_CLASSES:ClassVar = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    NAMES:ClassVar = {i: name for i, name in enumerate(VOC_CLASSES)}

    def __init__(
        self,
        root,
        year="2007",
        split="train",
        download=False,
        transforms=None,
        indices=None,
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=split,
            download=download,
            transforms=transforms,
        )

        self.transforms = transforms
        self.indices = list(range(len(self.images))) if indices is None else list(indices)

        self.image_ids = self.indices.copy()

        self.name = "VOC"
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[str, tuple[int, int], Any, dict[str, Any]]:
        real_index = self.indices[index]

        img_path = self.images[real_index]
        img = Image.open(img_path).convert("RGB")
        target = self.parse_voc_xml(
            ET_parse(self.annotations[real_index]).getroot(),
        )

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        image_size = img.size

        return img_path, image_size, img, target

    def shuffle(self):
        random.shuffle(self.indices)
        self.image_ids = self.indices.copy()

    def split_dataset(
        self,
        proportion,
        shuffle=False,
        n_calib_test: int|None = None,
    ):
        indices = self.indices.copy()

        if shuffle:
            logger.info("Shuffling dataset")
            random.shuffle(indices)

        n_total_samples = len(indices)
        if n_calib_test is not None:
            n_total_samples = min(n_total_samples, n_calib_test)

        indices = indices[:n_total_samples]
        n_split = int(proportion * n_total_samples)

        indices_1 = indices[:n_split]
        indices_2 = indices[n_split:]

        dataset_1 = VOCDataset(
            root=self.root,
            year=self.year,
            split=self.split,
            download=False,
            transforms=self.transforms,
            indices=indices_1,
        )

        dataset_2 = VOCDataset(
            root=self.root,
            year=self.year,
            split=self.split,
            download=False,
            transforms=self.transforms,
            indices=indices_2,
        )

        return dataset_1, dataset_2

    def _collate_fn(self, batch):
        return [list(x) for x in zip(*batch)]
