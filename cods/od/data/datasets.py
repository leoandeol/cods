import json
import os
import random
from logging import getLogger
from typing import Any, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

logger = getLogger("cods")

# from torchvision.datasets import VOCDetection, CocoDetection

# TODO: Add xview dataset
# TODO: Add voc dataset


class MSCOCODataset(Dataset):
    NAMES = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        12: "street sign",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        26: "hat",
        27: "backpack",
        28: "umbrella",
        29: "shoe",
        30: "eye glasses",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        45: "plate",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        66: "mirror",
        67: "dining table",
        68: "window",
        69: "desk",
        70: "toilet",
        71: "door",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        83: "blender",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
        91: "hair brush",
    }

    def __init__(self, root, split, transforms=None, image_ids=None):
        super().__init__()
        self.name = "MSCOCO"
        self.split = split
        self.root = root
        self.images_path = root  # os.path.join(self.path, "images")

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of ['train', 'val', 'test']")
        if split == "train":
            self.images_path = os.path.join(self.images_path, "train2017")
        elif split == "val":
            self.images_path = os.path.join(self.images_path, "val2017")
        elif split == "test":
            self.images_path = os.path.join(self.images_path, "test2017")
        self.annotations_path = os.path.join(self.root, "annotations")
        self._split = split
        if split == "train":
            self.annotations_path = os.path.join(
                self.annotations_path,
                "instances_train2017.json",
            )
        elif split == "val":
            self.annotations_path = os.path.join(
                self.annotations_path,
                "instances_val2017.json",
            )
        elif split == "test":
            raise ValueError("No annotations exists for the test set")
        # loading annotations
        self.annotations_json = json.load(open(self.annotations_path))

        # extract images ids
        if image_ids is None:
            self.image_ids = [
                image["id"] for image in self.annotations_json["images"]
            ]
        else:
            self.image_ids = image_ids
        self.image_files = {
            image["id"]: image["file_name"]
            for image in self.annotations_json["images"]
            if image["id"] in self.image_ids
        }
        self.annotations = self.annotations_json["annotations"]

        # rebuild annotations as a dictionary with the image_id as index and value as the set of annotations that are on this image
        self.reindexed_annotations = {
            image_id: [] for image_id in self.image_ids
        }
        for annotation in self.annotations:
            if annotation["image_id"] in self.image_ids:
                self.reindexed_annotations[annotation["image_id"]].append(
                    annotation,
                )

        self.transforms = transforms

    def __repr__(self):
        return f"MSCOCODataset(\n\t{self.name = },\n\t{self.split = },\n\t{self.root = }\n)"

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, idx: int):
        new_idx = self.image_ids[idx]
        image_path = os.path.join(self.images_path, self.image_files[new_idx])

        return Image.open(image_path)

    def _load_image_with_path(self, idx: int):
        new_idx = self.image_ids[idx]
        image_path = os.path.join(self.images_path, self.image_files[new_idx])
        return Image.open(image_path), image_path

    def _load_target(self, idx: int):
        new_idx = self.image_ids[idx]
        return self.reindexed_annotations[new_idx]

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        img, img_path = self._load_image_with_path(idx)
        target = self._load_target(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        image_size = img.size

        return img_path, image_size, img, target

    def __iter__(self):
        return iter(self.images_path)

    def __contains__(self, item):
        return item in self.images_path

    def shuffle(self):
        random.shuffle(self.image_ids)

    def split_dataset(
        self,
        proportion,
        shuffle=False,
        n_calib_test: Optional[int] = None,
    ):
        if shuffle:
            logger.info("Shuffling dataset")
            self.shuffle()

        n_total_samples = len(self)
        if n_calib_test is not None and n_calib_test < len(self):
            n_total_samples = n_calib_test

        n_split = int(proportion * n_total_samples)
        new_image_ids_1 = self.image_ids[:n_split]
        new_image_ids_2 = self.image_ids[n_split:n_total_samples]

        new_dataset_1 = MSCOCODataset(
            root=self.root,
            split=self._split,
            transforms=self.transforms,
            image_ids=new_image_ids_1,
        )
        new_dataset_2 = MSCOCODataset(
            root=self.root,
            split=self._split,
            transforms=self.transforms,
            image_ids=new_image_ids_2,
        )

        return new_dataset_1, new_dataset_2

    def _collate_fn(self, batch):
        return list([list(x) for x in zip(*batch)])
        # didn't check the above


class VOCDataset(VOCDetection):
    VOC_CLASSES = [
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

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """Args:
            index (int): Index

        Returns
        -------
            tuple: (image, target) where target is a dictionary of the XML tree.

        """
        img_path = self.images[index]
        img = Image.open(img_path).convert("RGB")
        target = self.parse_voc_xml(
            ET_parse(self.annotations[index]).getroot(),
        )

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        image_size = img.size

        return img_path, image_size, img, target
