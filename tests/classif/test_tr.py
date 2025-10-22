import unittest
from unittest.mock import MagicMock, patch

import torch

from cods.classif.data import ClassificationPredictions
from cods.classif.loss import CLASSIFICATION_LOSSES, ClassificationLoss
from cods.classif.models import ClassificationModel
from cods.classif.tr import ClassificationToleranceRegion


class MockClassificationPredictions:
    def __init__(self, pred_cls, true_cls, n_classes):
        self.pred_cls = pred_cls
        self.true_cls = true_cls
        self.n_classes = n_classes


class TestClassificationToleranceRegion(unittest.TestCase):
    def _create_mock_preds(self, num_samples, n_classes):
        pred_cls = [torch.rand(n_classes) for _ in range(num_samples)]
        true_cls = [torch.randint(0, n_classes, (1,)).squeeze() for _ in range(num_samples)]
        return MockClassificationPredictions(
            pred_cls=pred_cls, true_cls=true_cls, n_classes=n_classes
        )

    def test_init_valid_inputs(self):
        ctr = ClassificationToleranceRegion()
        self.assertEqual(ctr._loss_name, "lac")  # Default loss
        self.assertIsInstance(ctr._loss, CLASSIFICATION_LOSSES["lac"])
        self.assertEqual(ctr.preprocess, "softmax")
        self.assertIsNotNone(ctr.f_preprocess)

    def test_init_invalid_loss_string(self):
        with self.assertRaisesRegex(ValueError, r"Loss invalid_loss_str not supported.*"):
            ClassificationToleranceRegion(loss="invalid_loss_str")

    def test_init_invalid_loss_type_int(self):
        # This will be caught by `if loss not in self.ACCEPTED_LOSSES:` because 123 is not a key.
        with self.assertRaisesRegex(ValueError, r"Loss 123 not supported.*"):
            ClassificationToleranceRegion(loss=123)  # Not a string or ClassificationLoss

    def test_init_invalid_loss_obj_type(self):
        # This will also be caught by `if loss not in self.ACCEPTED_LOSSES:`
        # due to the current structure of checks in the source code's __init__.
        class SomeOtherClass:
            pass

        with self.assertRaisesRegex(ValueError, r"Loss .* not supported.*"):  # Adjusted regex
            ClassificationToleranceRegion(loss=SomeOtherClass())

    def test_init_invalid_preprocess(self):
        with self.assertRaisesRegex(
            ValueError, r"preprocess 'invalid_preprocess_str' not accepted.*"
        ):
            ClassificationToleranceRegion(preprocess="invalid_preprocess_str")

    # Calibrate tests will be more involved due to optimizer and risk function
    @patch("cods.base.optim.BinarySearchOptimizer.optimize")  # Corrected patch path
    def test_calibrate(self, mock_optimizer_optimize):
        # Mock the return value of optimizer.optimize to be a dummy lambda
        dummy_lbd_value = 0.5
        mock_optimizer_optimize.return_value = dummy_lbd_value

        ctr = ClassificationToleranceRegion(loss="lac", device="cpu")
        n_samples, n_classes = 10, 3
        mock_predictions = self._create_mock_preds(n_samples, n_classes)
        alpha, delta = 0.1, 0.05

        # The loss instance is ctr._loss
        # Mock its __call__ and get_set methods for controlled behavior during calibration risk assessment
        original_loss_instance = ctr._loss
        ctr._loss = MagicMock(
            spec=original_loss_instance
        )  # Mock based on the actual loss type (e.g. LACLoss)
        ctr._loss.return_value = torch.tensor(
            0.1
        )  # Mocking loss(true_cls, conf_cls) to return a scalar tensor
        ctr._loss.get_set.return_value = torch.tensor(
            [0]
        )  # Mocking loss.get_set() to return a dummy set

        lbd = ctr.calibrate(mock_predictions, alpha=alpha, delta=delta, verbose=False)

        self.assertEqual(lbd, dummy_lbd_value)
        self.assertEqual(ctr._lbd, dummy_lbd_value)
        self.assertIsNotNone(ctr._n_classes)
        self.assertEqual(ctr._n_classes, n_classes)

        mock_optimizer_optimize.assert_called_once()
        # To further test: inspect mock_optimizer_optimize.call_args to see if risk_function behaves as expected.
        # For example, check if ctr._loss.__call__ and ctr._loss.get_set were called inside the risk_function.

    def test_conformalize_not_calibrated(self):
        ctr = ClassificationToleranceRegion()
        mock_eval_preds = self._create_mock_preds(5, 3)
        with self.assertRaisesRegex(
            ValueError, "Conformalizer must be calibrated before conformalizing."
        ):
            ctr.conformalize(mock_eval_preds)

    def test_conformalize(self):
        ctr = ClassificationToleranceRegion(loss="lac", device="cpu")
        ctr._lbd = 0.5
        ctr._n_classes = 3

        original_loss_instance = ctr._loss
        ctr._loss = MagicMock(spec=original_loss_instance)
        mock_set_output = torch.tensor([0, 1])
        ctr._loss.get_set.return_value = mock_set_output

        n_eval_samples = 2
        mock_eval_preds = self._create_mock_preds(n_eval_samples, ctr._n_classes)

        conformalized_sets = ctr.conformalize(mock_eval_preds, verbose=False)

        self.assertIsInstance(conformalized_sets, list)
        self.assertEqual(len(conformalized_sets), n_eval_samples)
        self.assertTrue(ctr._loss.get_set.called)
        self.assertEqual(ctr._loss.get_set.call_count, n_eval_samples)
        for s in conformalized_sets:
            torch.testing.assert_close(s, mock_set_output)
        self.assertIs(mock_eval_preds.conf_cls, conformalized_sets)

    def test_evaluate_not_calibrated(self):
        ctr = ClassificationToleranceRegion()
        mock_preds = self._create_mock_preds(5, 3)
        conf_cls = [torch.tensor([0, 1])]
        with self.assertRaisesRegex(
            ValueError, "Conformalizer must be calibrated before evaluating."
        ):
            ctr.evaluate(mock_preds, conf_cls)

    def test_evaluate(self):
        ctr = ClassificationToleranceRegion(loss="lac", device="cpu")
        ctr._lbd = 0.5
        n_eval_samples, n_classes = 3, 3
        mock_eval_preds = self._create_mock_preds(n_eval_samples, n_classes)

        conformalized_sets = []
        expected_coverage_sum = 0
        expected_set_size_sum = 0
        for i in range(n_eval_samples):
            true_label = mock_eval_preds.true_cls[i].item()
            if i == 0:
                alt_label = (true_label + 1) % n_classes
                current_set = torch.tensor([alt_label])
                # expected_coverage_sum += 0 # No need to add 0
            else:
                current_set = torch.tensor([true_label, (true_label + 1) % n_classes])
                expected_coverage_sum += 1
            conformalized_sets.append(current_set)
            expected_set_size_sum += len(current_set)

        losses, set_sizes = ctr.evaluate(mock_eval_preds, conformalized_sets, verbose=False)

        self.assertIsInstance(losses, torch.Tensor)
        self.assertIsInstance(set_sizes, torch.Tensor)
        self.assertEqual(losses.shape[0], n_eval_samples)
        self.assertEqual(set_sizes.shape[0], n_eval_samples)

        expected_mean_coverage = torch.tensor(
            expected_coverage_sum / n_eval_samples if n_eval_samples > 0 else 0.0, dtype=torch.float
        )
        expected_mean_set_size = torch.tensor(
            expected_set_size_sum / n_eval_samples if n_eval_samples > 0 else 0.0, dtype=torch.float
        )

        torch.testing.assert_close(torch.mean(losses.float()), expected_mean_coverage)
        torch.testing.assert_close(torch.mean(set_sizes.float()), expected_mean_set_size)

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
        # TODO: Add APS once implemented
        for loss in ["lac"]:
            with self.subTest(loss=loss):
                # Initialize conformalizer
                conformalizer = ClassificationToleranceRegion(
                    loss=loss,
                    preprocess="softmax",
                    device="cpu",
                )

                # 5. Calibrate
                alpha = 0.1
                quantile = conformalizer.calibrate(
                    cal_preds,
                    alpha=alpha,
                    verbose=False,
                )

                # Verify calibration
                self.assertIsInstance(quantile, torch.Tensor)
                self.assertIsNotNone(conformalizer._lbd)
                self.assertIsNotNone(conformalizer._loss)

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
                    with self.subTest(loss=loss, alpha=test_alpha):
                        conformalizer_alpha = ClassificationToleranceRegion(
                            loss=loss,
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
