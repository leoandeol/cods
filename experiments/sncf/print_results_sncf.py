import logging
import os
import pickle
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cods.od.cp import ODConformalizer
from cods.od.models import YOLOModel

logging.getLogger().setLevel(logging.INFO)


class SNCFDataset(Dataset):
    """A PyTorch Dataset for the SNCF traffic light state dataset.

    This version correctly handles:
    1. Split files that contain full, absolute paths to images.
    2. Label files that contain multiple bounding boxes (one per line).
    """

    def __init__(
        self, root: str, split: str, transforms: Optional[callable] = None
    ):
        """Initializes the SNCFDataset.

        Args:
            root (str): The root directory of the dataset. This is used to find
                        the labels and classes.txt file.
            split (str): The name of the split to load (e.g., 'train', 'calib').
            transforms (callable, optional): A function/transform.

        """
        super().__init__()
        self.root = root
        self.split = split
        self.transforms = transforms

        # The labels directory is still found relative to the root
        self.labels_dir = os.path.join(self.root, "labels")
        self.classes_path = os.path.join(self.root, "classes.txt")

        self.classes = self._load_classes()

        # Load the split file which contains absolute paths to images
        split_file_path = os.path.join(self.root, f"{split}.txt")
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file not found: {split_file_path}")

        with open(split_file_path) as f:
            # self.image_files now contains absolute paths, e.g., '/path/to/images/file.jpg'
            self.image_files = [line.strip() for line in f if line.strip()]

        # image ids are just the file names witout extensions or path
        self.image_ids = [
            os.path.splitext(os.path.basename(img_path))[0]
            for img_path in self.image_files
        ]

        # TODO(leo): to satisfy rthe lib
        self.NAMES = self.classes

    def _load_classes(self) -> dict:
        """Loads class names from classes.txt."""
        with open(self.classes_path) as f:
            classes_raw = f.read().splitlines()
            classes_raw = [c.split(" ") for c in classes_raw if len(c) > 0]
        return {int(c[0]): " ".join(c[1:]) for c in classes_raw}

    def __len__(self) -> int:
        """Returns the total number of images in the dataset for the current split."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """Retrieves an item from the dataset at the specified index."""
        img_path = self.image_files[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # To find the label, we still need the base filename (e.g., 'image.jpg')
        base_img_filename = os.path.basename(img_path)
        label_name = os.path.splitext(base_img_filename)[0] + ".txt"

        # Construct the absolute path to the label file
        label_path = os.path.join(self.labels_dir, label_name)

        target = self._load_target(label_path, h, w)

        if self.transforms is not None:
            # Your transform function must accept both image and target
            # img, target = self.transforms(img, target)
            img = self.transforms(img)

        image_size = img.size

        return img_path, image_size, img, target

    def _load_target(self, label_path: str, height: int, width: int) -> dict:
        """Loads and processes a label file, handling multiple bounding boxes (lines)."""
        # --- MODIFICATION START ---
        # Initialize lists to hold all boxes and labels for a single image.
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            # Return empty lists if a label file doesn't exist for an image
            return {"boxes": boxes, "labels": labels}

        with open(label_path) as f:
            # Iterate over each line in the file, as each line is a new bounding box.
            for line in f:
                line = line.strip()
                if not line:
                    continue

                label_data = line.split(" ")

                # Basic validation for a valid YOLO format line
                if len(label_data) < 5:
                    continue

                class_id = label_data[0]
                box_coords = np.array([float(x) for x in label_data[1:5]])

                # Denormalize coordinates
                box_coords = box_coords * np.array(
                    [width, height, width, height]
                )

                # Convert (center_x, center_y, w, h) to (x_min, y_min, x_max, y_max)
                x_min = box_coords[0] - box_coords[2] / 2
                y_min = box_coords[1] - box_coords[3] / 2
                x_max = box_coords[0] + box_coords[2] / 2
                y_max = box_coords[1] + box_coords[3] / 2

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))  # self.classes[class_id])

        # The target dictionary now contains a list of boxes and a list of labels.
        return {"boxes": boxes, "labels": labels}
        # --- MODIFICATION END ---

    def _collate_fn(self, batch):
        return list([list(x) for x in zip(*batch)])


if __name__ == "__main__":
    name_of_pickle = "sncf_results.pkl"

    trans = transforms.Compose(
        [
            # transforms.ToTensor(), #
        ],
    )
    dataset_root_path = "/datasets/shared_datasets/SNCF/DATASET_etat_feu"

    # 1. Instantiate the training dataset
    # Used for training your model
    train_dataset = SNCFDataset(
        root=dataset_root_path, split="train", transforms=trans
    )

    # 2. Instantiate the validation dataset
    # Used for monitoring training and hyperparameter tuning
    val_dataset = SNCFDataset(
        root=dataset_root_path, split="val", transforms=trans
    )

    # 3. Instantiate the calibration dataset
    # A hold-out set used ONLY for calibrating your conformal predictor
    calib_dataset = SNCFDataset(
        root=dataset_root_path, split="calib", transforms=trans
    )

    # 4. Instantiate the test dataset
    # The final hold-out set used ONLY for evaluating the final, calibrated model
    test_dataset = SNCFDataset(
        root=dataset_root_path, split="test", transforms=trans
    )

    # --- Verification ---
    print("Dataset loading complete.")
    print("-" * 30)
    print(f"Training dataset size:   {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Calibration dataset size:{len(calib_dataset)}")
    print(f"Test dataset size:       {len(test_dataset)}")
    print("-" * 30)

    # Example of accessing a sample from the test set
    if len(test_dataset) > 0:
        img_path, _, _, target = test_dataset[0]
        print(f"\nFirst test sample path: {img_path}")
        print(f"First test sample target: {target}")

    MODELS = [
        ["cods-sncf/yolo11x_sncf4/weights/best.pt", 0.0263197],
        ["cods-sncf/yolov10x_sncf/weights/best.pt", 0],  # .01631],
        ["cods-sncf/yolo12x_sncf15/weights/best.pt", 0.02633],
        ["cods-sncf/yolo12x_sncf10/weights/last.pt", 0.026317],
        ["cods-sncf/yolo12x_sncf6/weights/best.pt", 0.026317],
    ]
    results = {}
    for model_name, thr in MODELS:
        try:
            model = YOLOModel(
                # model_name="./runs/detect/yolov8_sncf_augmented_training4/weights/best.pt",
                model_name=model_name,  # "cods-sncf/yolo11x_sncf4/weights/best.pt",
                pretrained=True,
                is_coco=False,
                # device="cuda:0",
            )
            bs = (
                18
                if model_name != "cods-sncf/yolo12x_sncf10/weights/last.pt"
                and model_name != "cods-sncf/yolo12x_sncf15/weights/best.pt"
                else (
                    6
                    if model_name != "cods-sncf/yolo12x_sncf15/weights/best.pt"
                    else 3
                )
            )
            logging.info(f"Loading model {model_name} with batch size {bs}")
            preds_cal = model.build_predictions(
                calib_dataset,
                dataset_name="sncf",
                split_name="cal",
                batch_size=bs,
                collate_fn=calib_dataset._collate_fn,  # TODO: make this a default for COCO
                shuffle=False,
                # force_recompute=True,
                deletion_method="nms",
                filter_preds_by_confidence=thr,  # 0,  # thr,
            )
            preds_test = model.build_predictions(
                test_dataset,
                dataset_name="sncf",
                split_name="test",
                batch_size=bs,
                collate_fn=test_dataset._collate_fn,
                shuffle=False,
                # force_recompute=True,
                deletion_method="nms",
                filter_preds_by_confidence=thr,  # 0,  # 3e-2,
            )
        except:
            logging.exception(f"Error loading model {model_name}. Skipping.")
            continue
        # import torch

        # print("--" * 30)
        # conf = torch.concatenate(preds_cal.confidences)
        # conf, _ = torch.sort(conf, descending=True)
        # q1000 = conf[1000]
        # q2000 = conf[2000]
        # q3000 = conf[3000] if len(conf) > 3000 else conf[-1]
        # logging.info(
        #     f"Confidences for {model_name}: 1000th: {q1000:.7f}, "
        #     f"2000th: {q2000:.7f}, 3000th: {q3000:.7f}",
        # )
        # continue
        conf = ODConformalizer(
            backend="auto",
            guarantee_level="image",
            matching_function="mix",
            multiple_testing_correction=None,
            confidence_method="box_count_recall",
            localization_method="pixelwise",
            localization_prediction_set="multiplicative",
            classification_method="binary",
            classification_prediction_set="lac",
            # device="cuda:0",
        )
        try:
            parameters = conf.calibrate(
                preds_cal,
                alpha_confidence=0.003,
                alpha_localization=0.005,
                alpha_classification=0.005,
            )

            conformal_preds = conf.conformalize(
                preds_test, parameters=parameters
            )

            results_test = conf.evaluate(
                preds_test,
                parameters=parameters,
                conformalized_predictions=conformal_preds,
                include_confidence_in_global=False,
            )
            results[model_name] = results_test
        except Exception as e:
            logging.exception(
                f"Error processing model {model_name}. Skipping."
            )
            results[model_name] = f"Error: {e!s}"
        finally:
            with open(name_of_pickle, "wb") as f:
                pickle.dump(results, f)
