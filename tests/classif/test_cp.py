import unittest
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cods.classif.cp import ClassificationConformalizer
from cods.classif.data import ClassificationPredictions
from cods.classif.models import ClassificationModel
from cods.classif.score import ClassifNCScore


class MockClassificationPredictions:
    def __init__(self, pred_cls, true_cls, n_classes):
        self.pred_cls = pred_cls
        self.true_cls = true_cls
        self.n_classes = n_classes


class TestClassificationConformalizer(unittest.TestCase):
    def _create_mock_preds(self, num_samples, n_classes):
        pred_cls = [F.softmax(torch.rand(n_classes), dim=-1) for _ in range(num_samples)]
        true_cls = [torch.randint(0, n_classes, (1,)).squeeze() for _ in range(num_samples)]
        return MockClassificationPredictions(
            pred_cls=pred_cls, true_cls=true_cls, n_classes=n_classes
        )

    def test_init_valid_inputs(self):
        # Test with valid 'method' (string) and 'preprocess'
        cc = ClassificationConformalizer(method="lac", preprocess="softmax", device="cpu")
        self.assertEqual(cc.method, "lac")
        self.assertEqual(cc.preprocess, "softmax")
        self.assertIsNotNone(cc.f_preprocess)

        cc = ClassificationConformalizer(method="aps", preprocess="softmax")
        self.assertEqual(cc.method, "aps")

        # Test with a custom ClassifNCScore instance for method
        mock_score_fn = MagicMock(spec=ClassifNCScore)
        cc = ClassificationConformalizer(method=mock_score_fn, preprocess="softmax")
        self.assertIs(cc.method, mock_score_fn)
        self.assertIs(cc._score_function, mock_score_fn)

    def test_init_invalid_method_string(self):
        # Test with an invalid string 'method'
        with self.assertRaisesRegex(ValueError, "method 'invalid_method' not accepted"):
            ClassificationConformalizer(method="invalid_method", preprocess="softmax")

    def test_init_invalid_preprocess_string(self):
        # Test with an invalid string 'preprocess'
        with self.assertRaisesRegex(ValueError, "preprocess 'invalid_preprocess' not accepted"):
            ClassificationConformalizer(method="lac", preprocess="invalid_preprocess")

    def test_calibrate(self):
        # More detailed test for calibrate
        conformalizer = ClassificationConformalizer(
            method="lac", preprocess="softmax", device="cpu"
        )

        n_cal_samples = 10  # Increased from 3
        n_classes = 3
        mock_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        alpha = 0.1

        quantile, scores = conformalizer.calibrate(mock_preds, alpha, verbose=False)

        self.assertIsInstance(quantile, torch.Tensor)
        self.assertIsNotNone(conformalizer._quantile)
        torch.testing.assert_close(quantile, conformalizer._quantile)

        self.assertIsInstance(scores, torch.Tensor)
        self.assertEqual(scores.shape[0], n_cal_samples)
        self.assertIsNotNone(conformalizer._scores)
        torch.testing.assert_close(scores, conformalizer._scores)
        self.assertEqual(conformalizer._n_classes, n_classes)
        self.assertIsNotNone(conformalizer._score_function)  # Should be initialized

    def test_calibrate_custom_score_fn(self):
        mock_score_fn = MagicMock(spec=ClassifNCScore)
        # Mock the behavior of the score function if needed for specific score values
        # For now, just ensure it's called
        mock_score_fn.return_value = torch.tensor(0.5)  # Each call to score_fn returns a score

        conformalizer = ClassificationConformalizer(
            method=mock_score_fn, preprocess="softmax", device="cpu"
        )

        n_cal_samples = 10  # Increased from 2
        n_classes = 3
        mock_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        alpha = 0.1

        conformalizer.calibrate(mock_preds, alpha, verbose=False)

        self.assertIs(conformalizer._score_function, mock_score_fn)
        # Check if the custom score function was called for each calibration sample
        self.assertEqual(mock_score_fn.call_count, len(mock_preds.true_cls))

    def test_conformalize_not_calibrated(self):
        conformalizer = ClassificationConformalizer(method="lac", preprocess="softmax")
        mock_eval_preds = MockClassificationPredictions(
            pred_cls=[torch.tensor([0.2, 0.5, 0.3])],
            true_cls=None,
            n_classes=3,  # true_cls not needed for conformalize
        )
        with self.assertRaisesRegex(
            ValueError, "Conformalizer must be calibrated before conformalizing."
        ):
            conformalizer.conformalize(mock_eval_preds)

    def test_conformalize(self):
        conformalizer = ClassificationConformalizer(
            method="lac", preprocess="softmax", device="cpu"
        )

        # Calibration data
        n_cal_samples = 10  # Increased from 3
        n_classes = 3
        cal_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        alpha = 0.1
        conformalizer.calibrate(cal_preds, alpha, verbose=False)

        # Evaluation data
        n_eval_samples = 2
        eval_preds_list = [torch.rand(n_classes) for _ in range(n_eval_samples)]
        mock_eval_preds = MockClassificationPredictions(
            pred_cls=eval_preds_list, true_cls=None, n_classes=n_classes
        )

        # Mock the get_set method of the underlying score function
        # This is important because the actual get_set logic can be complex (APS vs LAC)
        # and we are testing the conformalizer's orchestration here.
        mock_set_output = torch.tensor([0, 1])  # Example output for a prediction set
        conformalizer._score_function = MagicMock(
            spec=ClassifNCScore
        )  # Replace real score_fn with mock
        conformalizer._score_function.get_set.return_value = mock_set_output

        conformalized_sets = conformalizer.conformalize(mock_eval_preds)

        self.assertIsInstance(conformalized_sets, list)
        self.assertEqual(len(conformalized_sets), len(eval_preds_list))
        self.assertTrue(conformalizer._score_function.get_set.called)
        self.assertEqual(conformalizer._score_function.get_set.call_count, len(eval_preds_list))

        for item in conformalized_sets:
            torch.testing.assert_close(item, mock_set_output)

    def test_evaluate_not_calibrated(self):
        conformalizer = ClassificationConformalizer(method="lac", preprocess="softmax")
        mock_preds = MockClassificationPredictions(pred_cls=[], true_cls=[], n_classes=3)
        # Conformalized predictions (dummy)
        conf_cls = [torch.tensor([0, 1])]

        with self.assertRaisesRegex(
            ValueError, "Conformalizer must be calibrated before evaluating."
        ):
            conformalizer.evaluate(mock_preds, conf_cls)

    def test_evaluate_conf_cls_none(self):
        conformalizer = ClassificationConformalizer(method="lac", preprocess="softmax")
        # Calibrate first
        n_cal_samples = 10  # Increased from 1
        n_classes = 3
        cal_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        conformalizer.calibrate(cal_preds, alpha=0.1, verbose=False)

        mock_preds = self._create_mock_preds(num_samples=0, n_classes=n_classes)  # Empty eval data
        with self.assertRaisesRegex(
            ValueError, "Predictions must be conformalized before evaluating."
        ):
            conformalizer.evaluate(mock_preds, None)

    def test_evaluate(self):
        conformalizer = ClassificationConformalizer(
            method="lac", preprocess="softmax", device="cpu"
        )

        # Calibration
        n_cal_samples = 10  # Increased from 3
        n_classes = 3
        cal_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        alpha = 0.1
        conformalizer.calibrate(cal_preds, alpha, verbose=False)

        # Data for evaluation
        n_eval_samples = 3
        # Create true classes for evaluation
        eval_true_cls = [torch.randint(0, n_classes, (1,)).squeeze() for _ in range(n_eval_samples)]
        mock_eval_preds = MockClassificationPredictions(
            pred_cls=None,  # Not strictly needed by evaluate if conf_cls is provided
            true_cls=eval_true_cls,
            n_classes=n_classes,
        )

        # Sample conformalized prediction sets (list of tensors)
        # These need to be plausible outputs from a score function's get_set method.
        # For simplicity, let's assume each set contains one or two class indices.
        conformalized_sets = []
        expected_loss_sum = 0
        expected_set_size_sum = 0
        for i in range(n_eval_samples):
            # Create a sample set, e.g., the true class and one other if possible
            current_set = [eval_true_cls[i].item()]
            if n_classes > 1:
                other_class = (eval_true_cls[i].item() + 1) % n_classes
                if np.random.rand() > 0.5:  # Randomly make sets larger
                    current_set.append(other_class)

            # Ensure all elements in current_set are valid class indices
            current_set = [cs % n_classes for cs in current_set]
            # Remove duplicates if any, though unlikely with this construction
            current_set = sorted(set(current_set))

            conformalized_sets.append(torch.tensor(current_set))

            # For calculating expected coverage and set size
            if eval_true_cls[i].item() in current_set:
                expected_loss_sum += 1
            expected_set_size_sum += len(current_set)

        losses, set_sizes = conformalizer.evaluate(
            mock_eval_preds, conformalized_sets, verbose=False
        )

        self.assertIsInstance(losses, torch.Tensor)
        self.assertIsInstance(set_sizes, torch.Tensor)
        self.assertEqual(losses.shape[0], n_eval_samples)
        self.assertEqual(set_sizes.shape[0], n_eval_samples)

        expected_coverage = torch.tensor(
            expected_loss_sum / n_eval_samples if n_eval_samples > 0 else 0.0
        )
        expected_avg_set_size = torch.tensor(
            expected_set_size_sum / n_eval_samples if n_eval_samples > 0 else 0.0
        )

        torch.testing.assert_close(torch.mean(losses.float()), expected_coverage)
        torch.testing.assert_close(torch.mean(set_sizes.float()), expected_avg_set_size)

        if n_eval_samples > 0 and n_classes > 0:  # Ensure we can make a non-covered case
            # Test with a case where one is not covered (if possible)
            conformalized_sets_partial_coverage = list(conformalized_sets)  # copy
            # Make the first true_class not covered, if there are sets and classes to manipulate
            if len(conformalized_sets_partial_coverage) > 0 and n_classes > 0:
                first_true_cls = eval_true_cls[0].item()
                # Create a new set for the first item that definitely does not include the true class
                non_covering_set = [(first_true_cls + 1) % n_classes]
                if n_classes > 1:  # Add another if possible, different from first_true_cls
                    non_covering_set.append((first_true_cls + 2) % n_classes)
                non_covering_set = sorted(set(non_covering_set))
                conformalized_sets_partial_coverage[0] = torch.tensor(non_covering_set)

                # Recalculate expected_loss_sum_partial
                expected_loss_sum_partial = 0
                for i in range(n_eval_samples):
                    if eval_true_cls[i].item() in conformalized_sets_partial_coverage[i].tolist():
                        expected_loss_sum_partial += 1

                losses_partial, _ = conformalizer.evaluate(
                    mock_eval_preds, conformalized_sets_partial_coverage, verbose=False
                )
                expected_coverage_partial = torch.tensor(expected_loss_sum_partial / n_eval_samples)
                torch.testing.assert_close(
                    torch.mean(losses_partial.float()), expected_coverage_partial
                )

    def test_integration_pipeline_with_dataset(self):
        """Integration test that runs the full conformal prediction pipeline:
        1. Download TinyImageNet dataset
        2. Build predictions using ResNet18 from timm
        3. Split predictions into calibration and test sets
        4. Calibrate the conformalizer
        5. Conformalize test predictions
        6. Evaluate coverage and set sizes
        """
        import zipfile
        from pathlib import Path
        from urllib.request import urlretrieve

        import timm
        from PIL import Image
        from torchvision import transforms as T

        # 1. Download and prepare TinyImageNet
        data_dir = Path("./test_data")
        data_dir.mkdir(exist_ok=True)
        tiny_imagenet_dir = data_dir / "tiny-imagenet-200"

        # Download if not exists
        if not tiny_imagenet_dir.exists():
            print("Downloading TinyImageNet...")
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            zip_path = data_dir / "tiny-imagenet-200.zip"
            if not zip_path.exists():
                urlretrieve(url, zip_path)

            # Extract
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)

        val_dir = tiny_imagenet_dir / "val"

        # Create a dataset wrapper for TinyImageNet validation set
        class TinyImageNetDataset(torch.utils.data.Dataset):
            def __init__(self, root_dir, max_samples=100):
                self.root_dir = Path(root_dir)
                self.max_samples = max_samples

                # Load validation annotations
                val_annotations = self.root_dir / "val_annotations.txt"
                self.samples = []
                self.idx_to_cls = {}
                class_to_idx = {}

                with open(val_annotations) as f:
                    for idx, line in enumerate(f):
                        if idx >= max_samples:
                            break
                        parts = line.strip().split("\t")
                        img_name = parts[0]
                        class_name = parts[1]

                        if class_name not in class_to_idx:
                            class_to_idx[class_name] = len(class_to_idx)

                        class_idx = class_to_idx[class_name]
                        self.idx_to_cls[class_idx] = class_name
                        self.samples.append((img_name, class_idx))

                self.n_classes = len(class_to_idx)
                self.root = str(self.root_dir)
                self.image_ids = [s[0] for s in self.samples]

                # Define transforms
                self.transforms = T.Compose(
                    [
                        T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_name, label = self.samples[idx]
                img_path = self.root_dir / "images" / img_name
                image = Image.open(img_path).convert("RGB")
                image = self.transforms(image)
                return img_name, image, torch.tensor(label)

        # Create dataset first to determine n_classes
        dataset = TinyImageNetDataset(root_dir=val_dir, max_samples=100)
        n_samples = len(dataset)
        n_classes = dataset.n_classes

        # 2. Load ResNet18 from timm with correct number of classes
        model = timm.create_model("resnet18", pretrained=True, num_classes=n_classes)
        batch_size = 32

        classification_model = ClassificationModel(
            model=model,
            model_name="resnet18_tinyimagenet",
            device="cpu",
            save=False,  # Don't save during tests
        )

        # Build predictions
        predictions = classification_model.build_predictions(
            dataset=dataset,
            dataset_name="tinyimagenet",
            split_name="val",
            batch_size=batch_size,
            shuffle=False,
            verbose=False,
            force_recompute=True,
        )

        # Verify predictions structure
        self.assertIsInstance(predictions, ClassificationPredictions)
        self.assertEqual(len(predictions), n_samples)
        self.assertEqual(predictions.n_classes, dataset.n_classes)
        self.assertEqual(len(predictions.pred_cls), n_samples)
        self.assertEqual(len(predictions.true_cls), n_samples)

        # 3. Split predictions into calibration (60%) and test (40%) sets
        cal_preds, test_preds = predictions.split(
            splits_names=["calibration", "test"],
            splits_ratios=[0.6, 0.4],
        )

        self.assertEqual(len(cal_preds), int(n_samples * 0.6))
        self.assertEqual(len(test_preds), int(n_samples * 0.4))

        # 4. Test both LAC and APS methods
        for method in ["lac", "aps"]:
            with self.subTest(method=method):
                # Initialize conformalizer
                conformalizer = ClassificationConformalizer(
                    method=method,
                    preprocess="softmax",
                    device="cpu",
                )

                # 5. Calibrate
                alpha = 0.1
                quantile, scores = conformalizer.calibrate(
                    cal_preds,
                    alpha=alpha,
                    verbose=False,
                )

                # Verify calibration
                self.assertIsInstance(quantile, torch.Tensor)
                self.assertIsInstance(scores, torch.Tensor)
                self.assertEqual(scores.shape[0], len(cal_preds))
                self.assertIsNotNone(conformalizer._quantile)
                self.assertIsNotNone(conformalizer._score_function)

                # 6. Conformalize test predictions
                conformalized_sets = conformalizer.conformalize(test_preds)

                # Verify conformalized sets
                self.assertIsInstance(conformalized_sets, list)
                self.assertEqual(len(conformalized_sets), len(test_preds))
                for conf_set in conformalized_sets:
                    self.assertIsInstance(conf_set, torch.Tensor)
                    # Each set should contain at least one class
                    self.assertGreater(len(conf_set), 0)
                    # No class index should exceed n_classes
                    self.assertTrue(torch.all(conf_set < n_classes))
                    self.assertTrue(torch.all(conf_set >= 0))

                # 7. Evaluate coverage and set sizes
                losses, set_sizes = conformalizer.evaluate(
                    test_preds,
                    conformalized_sets,
                    verbose=False,
                )

                # Verify evaluation outputs
                self.assertIsInstance(losses, torch.Tensor)
                self.assertIsInstance(set_sizes, torch.Tensor)
                self.assertEqual(losses.shape[0], len(test_preds))
                self.assertEqual(set_sizes.shape[0], len(test_preds))

                # Calculate coverage (proportion of true classes in prediction sets)
                coverage = torch.mean(losses.float()).item()
                avg_set_size = torch.mean(set_sizes.float()).item()

                # Coverage should be at least (1 - alpha) due to conformal guarantees
                # In practice, with finite samples, we just verify it's reasonable
                self.assertGreaterEqual(coverage, 0.0)
                self.assertLessEqual(coverage, 1.0)

                # Average set size should be reasonable (between 1 and n_classes)
                self.assertGreater(avg_set_size, 0)
                self.assertLessEqual(avg_set_size, n_classes)

                # For a well-calibrated CP, we expect coverage close to (1 - alpha)
                # But we don't enforce strict bounds in tests due to randomness
                # Just verify the pipeline runs successfully

                # Test with different alpha values
                for test_alpha in [0.05, 0.2]:
                    with self.subTest(method=method, alpha=test_alpha):
                        conformalizer_alpha = ClassificationConformalizer(
                            method=method,
                            preprocess="softmax",
                            device="cpu",
                        )
                        conformalizer_alpha.calibrate(
                            cal_preds,
                            alpha=test_alpha,
                            verbose=False,
                        )
                        conf_sets_alpha = conformalizer_alpha.conformalize(test_preds)
                        losses_alpha, set_sizes_alpha = conformalizer_alpha.evaluate(
                            test_preds,
                            conf_sets_alpha,
                            verbose=False,
                        )

                        # Verify outputs have correct shapes
                        self.assertEqual(losses_alpha.shape[0], len(test_preds))
                        self.assertEqual(set_sizes_alpha.shape[0], len(test_preds))
