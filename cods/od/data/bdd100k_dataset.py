import json
import random
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from torch.utils.data import Dataset

logger = getLogger("cods")


class BDD100KDataset(Dataset):
    BDD_CLASSES: ClassVar[list[str]] = [
        "bike",
        "bus",
        "car",
        "motor",
        "person",
        "rider",
        "traffic light",
        "traffic sign",
        "train",
        "truck",
    ]

    NAMES: ClassVar[dict[int, str]] = {
        i: name for i, name in enumerate(BDD_CLASSES)
    }

    def __init__(
        self,
        root: str,
        split: str,
        transforms=None,
        image_ids: list[int] | None = None,
        keep_empty: bool = True,
    ):
        super().__init__()

        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")

        self.root = Path(root)
        self.split = split
        self.name = "BDD100K"
        self.transforms = transforms
        self.keep_empty = keep_empty

        self.images_path = self.root / "100k" / split
        if not self.images_path.is_dir():
            raise RuntimeError(f"Image directory not found: {self.images_path}")

        self.image_files = sorted(self.images_path.glob("*.jpg"))

        print(f"nb images : {len(self.image_files)}")

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_path}")

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.BDD_CLASSES)
        }

        if image_ids is None:
            self.image_ids = list(range(len(self.image_files)))
        else:
            self.image_ids = list(image_ids)

        if not keep_empty:
            self.image_ids = [
                idx for idx in self.image_ids
                if len(self._load_target_from_image_path(self.image_files[idx])) > 0
            ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image_with_path(self, idx: int) -> tuple[Image.Image, Path]:
        real_idx = self.image_ids[idx]
        image_path = self.image_files[real_idx]
        image = Image.open(image_path).convert("RGB")
        return image, image_path

    def _load_json(self, image_path: Path) -> dict[str, Any]:
        json_path = image_path.with_suffix(".json")

        if not json_path.is_file():
            return {}

        with open(json_path, "r") as f:
            return json.load(f)

    def _load_target_from_image_path(self, image_path: Path) -> list[dict[str, Any]]:
        data = self._load_json(image_path)

        frames = data.get("frames", [])
        if len(frames) == 0:
            return []

        objects = frames[0].get("objects", [])

        annotations: list[dict[str, Any]] = []

        for obj in objects:
            category = obj.get("category")
            box2d = obj.get("box2d")
            if box2d is None:
                continue

            if category not in self.class_to_idx:
                continue

            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])

            if x2 <= x1 or y2 <= y1:
                continue

            annotations.append(
                {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "category_id": self.class_to_idx[category],
                    "category_name": category,
                    "attributes": obj.get("attributes", {}),
                    "id": obj.get("id"),
                }
            )

        return annotations

    def _load_target(self, idx: int) -> list[dict[str, Any]]:
        real_idx = self.image_ids[idx]
        image_path = self.image_files[real_idx]
        return self._load_target_from_image_path(image_path)

    def __getitem__(self, idx: int) -> tuple[str, tuple[int, int], Any, list[dict[str, Any]]]:
        image, image_path = self._load_image_with_path(idx)

        image_size = image.size

        target = self._load_target_from_image_path(image_path)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return str(image_path), image_size, image, target

    def shuffle(self) -> None:
        random.shuffle(self.image_ids)

    def split_dataset(
        self,
        proportion: float,
        shuffle: bool = False,
        n_calib_test: int | None = None,
    ) -> tuple["BDD100KDataset", "BDD100KDataset"]:
        if not 0.0 <= proportion <= 1.0:
            raise ValueError("proportion must be between 0 and 1")

        indices = self.image_ids.copy()

        if shuffle:
            logger.info("Shuffling dataset")
            random.shuffle(indices)

        n_total_samples = len(indices)
        if n_calib_test is not None:
            n_total_samples = min(n_total_samples, n_calib_test)

        indices = indices[:n_total_samples]
        n_split = int(proportion * n_total_samples)

        dataset_1 = BDD100KDataset(
            root=str(self.root),
            split=self.split,
            transforms=self.transforms,
            image_ids=indices[:n_split],
            keep_empty=self.keep_empty,
        )

        dataset_2 = BDD100KDataset(
            root=str(self.root),
            split=self.split,
            transforms=self.transforms,
            image_ids=indices[n_split:],
            keep_empty=self.keep_empty,
        )

        return dataset_1, dataset_2

    def _collate_fn(self, batch):
        return [list(x) for x in zip(*batch)]