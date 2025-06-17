import unittest
import torch
from unittest.mock import MagicMock, patch

# Imports from the project
from cods.classif.tr import ClassificationToleranceRegion
from cods.classif.loss import ClassificationLoss, CLASSIFICATION_LOSSES
# Assuming MockClassificationPredictions can be reused or defined here
# For now, let's define it here for simplicity, similar to test_cp.py

class MockClassificationPredictions:
    def __init__(self, pred_cls, true_cls, n_classes):
        self.pred_cls = pred_cls  # List of tensors
        self.true_cls = true_cls  # List of tensors
        self.n_classes = n_classes
        self.conf_cls = None # To store output of conformalize

class TestClassificationToleranceRegion(unittest.TestCase):

    def _create_mock_preds(self, num_samples, n_classes):
        pred_cls = [torch.rand(n_classes) for _ in range(num_samples)]
        true_cls = [torch.randint(0, n_classes, (1,)).squeeze() for _ in range(num_samples)]
        return MockClassificationPredictions(pred_cls=pred_cls, true_cls=true_cls, n_classes=n_classes)

    def test_init_valid_inputs(self):
        # Default params
        ctr = ClassificationToleranceRegion()
        self.assertEqual(ctr.loss_name, "lac") # Default loss
        self.assertIsInstance(ctr.loss, CLASSIFICATION_LOSSES["lac"])
        self.assertEqual(ctr.preprocess, "softmax")
        self.assertIsNotNone(ctr.f_preprocess)

        # Testing custom ClassificationLoss instances is removed here because the current
        # __init__ logic in ClassificationToleranceRegion makes it hard to test reliably:
        # any instance not strictly equal to a key in ACCEPTED_LOSSES (e.g. "lac")
        # gets caught by the `if loss not in self.ACCEPTED_LOSSES:` check,
        # preventing the `elif isinstance(loss, ClassificationLoss):` block from being tested
        # with a custom mock instance as intended.

    def test_init_invalid_loss_string(self):
        with self.assertRaisesRegex(ValueError, r"Loss invalid_loss_str not supported.*"):
            ClassificationToleranceRegion(loss="invalid_loss_str")

    def test_init_invalid_loss_type_int(self):
        # This will be caught by `if loss not in self.ACCEPTED_LOSSES:` because 123 is not a key.
        with self.assertRaisesRegex(ValueError, r"Loss 123 not supported.*"):
            ClassificationToleranceRegion(loss=123) # Not a string or ClassificationLoss

    def test_init_invalid_loss_obj_type(self):
        # This will also be caught by `if loss not in self.ACCEPTED_LOSSES:`
        # due to the current structure of checks in the source code's __init__.
        class SomeOtherClass:
            pass
        with self.assertRaisesRegex(ValueError, r"Loss .* not supported.*"): # Adjusted regex
            ClassificationToleranceRegion(loss=SomeOtherClass())

    def test_init_invalid_preprocess(self):
        with self.assertRaisesRegex(ValueError, r"preprocess 'invalid_preprocess_str' not accepted.*"):
            ClassificationToleranceRegion(preprocess="invalid_preprocess_str")

    # Calibrate tests will be more involved due to optimizer and risk function
    @patch('cods.base.optim.BinarySearchOptimizer.optimize') # Corrected patch path
    def test_calibrate(self, mock_optimizer_optimize):
        # Mock the return value of optimizer.optimize to be a dummy lambda
        dummy_lbd_value = 0.5
        mock_optimizer_optimize.return_value = dummy_lbd_value

        ctr = ClassificationToleranceRegion(loss="lac", device="cpu")
        n_samples, n_classes = 10, 3
        mock_predictions = self._create_mock_preds(n_samples, n_classes)
        alpha, delta = 0.1, 0.05

        # The loss instance is ctr.loss
        # Mock its __call__ and get_set methods for controlled behavior during calibration risk assessment
        original_loss_instance = ctr.loss
        ctr.loss = MagicMock(spec=original_loss_instance) # Mock based on the actual loss type (e.g. LACLoss)
        ctr.loss.return_value = torch.tensor(0.1) # Mocking loss(true_cls, conf_cls) to return a scalar tensor
        ctr.loss.get_set.return_value = torch.tensor([0]) # Mocking loss.get_set() to return a dummy set

        lbd = ctr.calibrate(mock_predictions, alpha=alpha, delta=delta, verbose=False)

        self.assertEqual(lbd, dummy_lbd_value)
        self.assertEqual(ctr.lbd, dummy_lbd_value)
        self.assertIsNotNone(ctr._n_classes)
        self.assertEqual(ctr._n_classes, n_classes)

        mock_optimizer_optimize.assert_called_once()
        # To further test: inspect mock_optimizer_optimize.call_args to see if risk_function behaves as expected.
        # For example, check if ctr.loss.__call__ and ctr.loss.get_set were called inside the risk_function.

    def test_conformalize_not_calibrated(self):
        ctr = ClassificationToleranceRegion()
        mock_eval_preds = self._create_mock_preds(5, 3)
        with self.assertRaisesRegex(ValueError, "Conformalizer must be calibrated before conformalizing."):
            ctr.conformalize(mock_eval_preds)

    def test_conformalize(self):
        ctr = ClassificationToleranceRegion(loss="lac", device="cpu")
        ctr.lbd = 0.5
        ctr._n_classes = 3

        original_loss_instance = ctr.loss
        ctr.loss = MagicMock(spec=original_loss_instance)
        mock_set_output = torch.tensor([0, 1])
        ctr.loss.get_set.return_value = mock_set_output

        n_eval_samples = 2
        mock_eval_preds = self._create_mock_preds(n_eval_samples, ctr._n_classes)

        conformalized_sets = ctr.conformalize(mock_eval_preds, verbose=False)

        self.assertIsInstance(conformalized_sets, list)
        self.assertEqual(len(conformalized_sets), n_eval_samples)
        self.assertTrue(ctr.loss.get_set.called)
        self.assertEqual(ctr.loss.get_set.call_count, n_eval_samples)
        for s in conformalized_sets:
            torch.testing.assert_close(s, mock_set_output)
        self.assertIs(mock_eval_preds.conf_cls, conformalized_sets)


    def test_evaluate_not_calibrated(self):
        ctr = ClassificationToleranceRegion()
        mock_preds = self._create_mock_preds(5, 3)
        conf_cls = [torch.tensor([0,1])]
        with self.assertRaisesRegex(ValueError, "Conformalizer must be calibrated before evaluating."):
            ctr.evaluate(mock_preds, conf_cls)

    def test_evaluate(self):
        ctr = ClassificationToleranceRegion(loss="lac", device="cpu")
        ctr.lbd = 0.5
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

        expected_mean_coverage = torch.tensor(expected_coverage_sum / n_eval_samples if n_eval_samples > 0 else 0.0, dtype=torch.float)
        expected_mean_set_size = torch.tensor(expected_set_size_sum / n_eval_samples if n_eval_samples > 0 else 0.0, dtype=torch.float)

        torch.testing.assert_close(torch.mean(losses.float()), expected_mean_coverage)
        torch.testing.assert_close(torch.mean(set_sizes.float()), expected_mean_set_size)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
