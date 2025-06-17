import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO

from cods.od.models.model import ODModel


def xywh2xyxy_scaled(x, width_scale, height_scale):
    """Convert bounding boxes from center (x, y, w, h) format to (x0, y0, x1, y1) format and scale.

    Args:
    ----
        x (torch.Tensor): Bounding boxes in (center_x, center_y, width, height) format.
        width_scale (float): Scaling factor for width.
        height_scale (float): Scaling factor for height.

    Returns:
    -------
        torch.Tensor: Bounding boxes in (x0, y0, x1, y1) format, scaled.

    """
    y = x.clone()
    y[:, 0] = (x[:, 0] - x[:, 2] / 2) * width_scale  # top left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * height_scale  # top left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2) * width_scale  # bottom right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * height_scale  # bottom right y
    return y


class AlteredYOLO(YOLO):
    """YOLO model wrapper with hooks to capture raw outputs and input shapes during prediction."""

    def __init__(self, model_path):
        """Initialize the AlteredYOLO model.

        Args:
        ----
            model_path (str): Path to the YOLO model weights.

        """
        super().__init__(model_path, verbose=False)
        self.raw_output = None

    def predict(self, source=None, stream=False, **kwargs):
        """Run prediction and capture raw outputs and input shapes using hooks.

        Args:
        ----
            source: Input source for prediction.
            stream (bool, optional): Whether to stream results. Defaults to False.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
        -------
            The results from YOLO prediction.

        """

        def output_hook(module, input, output):
            self.raw_output = output[0].clone()

        def image_hook(module, input, output):
            # print(input[0].shape[2:][::-1])
            self.input_shape = input[0].shape[2:][::-1]

        # Register the forward hook
        start_hook = self.model.model[0].register_forward_hook(image_hook)
        end_hook = self.model.model[-1].register_forward_hook(output_hook)

        # Run prediction
        results = super().predict(source, stream, verbose=False, **kwargs)

        # Remove the hook
        start_hook.remove()
        end_hook.remove()

        return results


class YOLOModel(ODModel):
    """Object Detection model wrapper for YOLO with custom preprocessing and postprocessing."""

    def __init__(
        self,
        model_name="yolov8x.pt",
        pretrained=True,
        weights=None,
        device="cpu",
        save=True,
        save_dir_path=None,
    ):
        """Initialize the YOLOModel.

        Args:
        ----
            model_name (str, optional): Name or path of the YOLO model. Defaults to "yolov8x.pt".
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
            weights: Custom weights (not used currently).
            device (str, optional): Device to use. Defaults to "cpu".
            save (bool, optional): Whether to save the model. Defaults to True.
            save_dir_path (str, optional): Directory to save the model. Defaults to None.

        """
        super().__init__(
            model_name=model_name,
            save_dir_path=save_dir_path,
            pretrained=pretrained,
            weights=weights,
            device=device,
        )
        self._model_name = model_name
        if pretrained is True:
            self.model = AlteredYOLO(model_name)
        else:
            raise NotImplementedError(
                "Only pretrained models are available for now",
            )
        self.device = device
        # self.model.eval()
        self.model.to(device)
        # TODO debug
        self.transform = T.Compose(
            [
                # T.Resize(800),
                # T.ToTensor(),
                # ResizeChannels(3),
                # T.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ],
        )

    # Unsure if this is the right way to do it, there is different ways to define the softmax
    def postprocess(
        self,
        raw_output,
        img_shapes,
        model_input_size,
    ):
        """Postprocess raw model outputs to obtain bounding boxes, confidences, and class probabilities.

        Args:
        ----
            raw_output (list[torch.Tensor]): Raw outputs from the model.
            img_shapes (torch.FloatTensor): Original image shapes.
            model_input_size (tuple): Model input size (width, height).

        Returns:
        -------
            tuple: (all_boxes, all_confs, all_probs) for each image in the batch.

        """
        all_boxes = []
        all_confs = []
        all_probs = []
        model_width, model_height = model_input_size
        for i in range(len(raw_output)):
            original_width, original_height = img_shapes[i]

            box_output = raw_output[i].t()

            # Calculate scaling factors
            width_scale = original_width / model_width
            height_scale = original_height / model_height

            # convert to [x0, y0, x1, y1] format
            out_boxes = box_output[:, :4]
            boxes = xywh2xyxy_scaled(out_boxes, width_scale, height_scale)

            temp = 1  # 0.2
            cls_probs = torch.softmax(box_output[:, 4:] / temp, dim=-1)

            # Extend to be of size 91 :
            CONVERT_TO_91 = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                27,
                28,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                67,
                70,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
            ]
            CONVERT_TO_91 = torch.tensor(
                CONVERT_TO_91,
                device=cls_probs.device,
            )

            cls_probs_new = torch.zeros(
                cls_probs.shape[0],
                91,
                device=cls_probs.device,
            )

            cls_probs_new[:, CONVERT_TO_91] = cls_probs

            final_confidence, predicted_class = torch.max(
                box_output[:, 4:],
                dim=-1,
            )
            # print(cls_probs_new.sum(dim=-1)[torch.abs(cls_probs_new.sum(dim=-1)-1)>1e-9])
            # for i in range(cls_probs_new.shape[0]):
            #    if torch.abs(cls_probs_new[i].sum()-1)>1e-9:
            #        print(f"{cls_probs_new[i].sum():10f} : {torch.abs(cls_probs_new[i].sum()-1)}")
            # print("\n")
            assert (torch.abs(cls_probs_new.sum(dim=-1) - 1) < 1e-6).all()

            all_boxes.append(boxes)
            all_confs.append(final_confidence)
            all_probs.append(cls_probs_new)

        return all_boxes, all_confs, all_probs

    def predict_batch(self, batch: list, **kwargs) -> dict:
        """Predict the output given a batch of input tensors.

        Args:
        ----
            batch (list): The input batch containing image paths, image sizes, images, and ground truth.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
        -------
            dict: The predicted output as a dictionary with the following keys:
                - "image_paths" (list): The paths of the input images
                - "image_shapes" (list): The shapes of the input images
                - "true_boxes" (list): The true bounding boxes of the objects in the images
                - "pred_boxes" (list): The predicted bounding boxes of the objects in the images
                - "confidences" (list): The confidence scores of the predicted bounding boxes
                - "true_cls" (list): The true class labels of the objects in the images
                - "pred_cls" (list): The predicted class labels of the objects in the images

        """
        image_paths, image_sizes, images, ground_truth = batch

        img_shapes = torch.FloatTensor(
            np.stack([image.size for image in images]),
        ).to(self.device)

        images = [self.transform(image) for image in images]
        # images = list([image.to(self.device) for image in images])
        self.model(images)
        raw_output = self.model.raw_output
        model_input_size = self.model.input_shape
        pred_boxes, confidences, pred_cls = self.postprocess(
            raw_output,
            img_shapes,
            model_input_size,
        )
        true_boxes = [
            torch.LongTensor(
                [
                    [
                        box["bbox"][0],
                        box["bbox"][1],
                        box["bbox"][0] + box["bbox"][2],
                        box["bbox"][1] + box["bbox"][3],
                    ]
                    for box in true_box
                ],
            )
            for true_box in ground_truth
        ]
        true_cls = [
            torch.LongTensor([box["category_id"] for box in true_box]) for true_box in ground_truth
        ]
        true_boxes = true_boxes

        return {
            "image_paths": image_paths,
            "image_shapes": image_sizes,
            "true_boxes": true_boxes,
            "pred_boxes": pred_boxes,
            "confidences": confidences,
            "true_cls": true_cls,
            "pred_cls": pred_cls,
        }
