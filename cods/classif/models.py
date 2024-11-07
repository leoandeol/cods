import torch
from tqdm import tqdm

from cods.base.models import Model
from cods.classif.data import ClassificationPredictions


class ClassificationModel(Model):
    def __init__(
        self,
        model,
        model_name,
        pretrained=True,
        weights=None,
        device="cpu",
        save=True,
        save_dir_path=None,
    ):
        super().__init__(
            model_name=model_name,
            save_dir_path=save_dir_path,
            pretrained=pretrained,
            weights=weights,
            device=device,
        )
        self.model = model.to(device)
        self.model.eval()

    def build_predictions(
        self,
        dataset,
        dataset_name: str,
        split_name: str,
        batch_size: int,
        shuffle: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        preds = self._load_preds_if_exists(
            dataset_name=dataset_name,
            split_name=split_name,
            task_name="classification",
        )
        if preds is not None:
            if verbose:
                print("Predictions already exist, loading them...")
            return preds
        elif verbose:
            print("Predictions do not exist, building them...")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
        )
        predictions = {"true_cls": [], "pred_cls": []}
        if verbose:
            print("Building predictions...")
        ids = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, disable=not verbose)):
                id, images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                ids.extend(id)
                preds = self.model(images)
                # _, preds = torch.max(outputs, 1)
                predictions["true_cls"].extend(labels)  # .cpu().numpy())
                predictions["pred_cls"].extend(preds)  # .cpu().numpy())
        predictions["dataset_name"] = dataset_name
        predictions["split_name"] = split_name
        paths = ids  # dataset.get_paths(ids)
        predictions["image_paths"] = paths
        predictions["idx_to_cls"] = dataset.idx_to_cls
        predictions["pred_cls"] = torch.stack(predictions["pred_cls"])
        predictions["true_cls"] = torch.stack(predictions["true_cls"])
        preds = ClassificationPredictions(**predictions)
        self._save_preds(preds)
        return preds
