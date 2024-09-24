import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset

# from torchvision.datasets import VOCDetection, CocoDetection

# TODO: Add xview dataset
# TODO: Add voc dataset


class MSCOCODataset(Dataset):
    NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }

    def __init__(self, root, split, transforms=None, image_ids=None, **kwargs):
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
                self.annotations_path, "instances_train2017.json"
            )
        elif split == "val":
            self.annotations_path = os.path.join(
                self.annotations_path, "instances_val2017.json"
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
                    annotation
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

    def __getitem__(self, idx: int):
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

    from typing import Optional

    # split dataset in two dataset randomly
    def random_split(
        self,
        proportion,
        shuffle=True,
        n_calib_test: Optional[int] = None,
        **kwargs,
    ):
        if shuffle:
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
            **kwargs,
        )
        new_dataset_2 = MSCOCODataset(
            root=self.root,
            split=self._split,
            transforms=self.transforms,
            image_ids=new_image_ids_2,
            **kwargs,
        )

        return new_dataset_1, new_dataset_2

    def _collate_fn(self, batch):
        return list([list(x) for x in zip(*batch)])
        # didn't check the above
