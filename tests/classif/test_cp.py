import unittest
import torch
import numpy as np # Will keep for potential raw data generation before converting to tensor
from unittest.mock import MagicMock, PropertyMock

from cods.classif.cp import ClassificationConformalizer
from cods.classif.score import ClassifNCScore # For type checking custom method

# Mock for ClassificationPredictions
class MockClassificationPredictions:
    def __init__(self, pred_cls, true_cls, n_classes):
        self.pred_cls = pred_cls # List of tensors
        self.true_cls = true_cls # List of tensors
        self.n_classes = n_classes
        # Mock the device attribute if methods in ClassificationConformalizer use it directly from preds
        # For example: self.device = torch.device("cpu")


class TestClassificationConformalizer(unittest.TestCase):

    def _create_mock_preds(self, num_samples, n_classes):
        # Ensure pred_cls and true_cls are lists of tensors
        pred_cls = [torch.rand(n_classes) for _ in range(num_samples)]
        # Ensure true_cls are scalar tensors (long type for class indices)
        true_cls = [torch.randint(0, n_classes, (1,)).squeeze() for _ in range(num_samples)]
        return MockClassificationPredictions(pred_cls=pred_cls, true_cls=true_cls, n_classes=n_classes)

    def test_init_valid_inputs(self):
        # Test with valid 'method' (string) and 'preprocess'
        cc = ClassificationConformalizer(method='lac', preprocess='softmax', device='cpu')
        self.assertEqual(cc.method, 'lac')
        self.assertEqual(cc.preprocess, 'softmax')
        self.assertIsNotNone(cc.f_preprocess)

        cc = ClassificationConformalizer(method='aps', preprocess='softmax')
        self.assertEqual(cc.method, 'aps')

        # Test with a custom ClassifNCScore instance for method
        mock_score_fn = MagicMock(spec=ClassifNCScore)
        cc = ClassificationConformalizer(method=mock_score_fn, preprocess='softmax')
        self.assertIs(cc.method, mock_score_fn)
        self.assertIs(cc._score_function, mock_score_fn)


    def test_init_invalid_method_string(self):
        # Test with an invalid string 'method'
        with self.assertRaisesRegex(ValueError, "method 'invalid_method' not accepted"):
            ClassificationConformalizer(method='invalid_method', preprocess='softmax')

    def test_init_invalid_preprocess_string(self):
        # Test with an invalid string 'preprocess'
        with self.assertRaisesRegex(ValueError, "preprocess 'invalid_preprocess' not accepted"):
            ClassificationConformalizer(method='lac', preprocess='invalid_preprocess')

    # It seems preprocess only accepts string keys, not direct functions or other types.
    # The original test_init_invalid_preprocess tested for a non-string type,
    # but the class logic checks `preprocess not in self.ACCEPTED_PREPROCESS.keys()`,
    # which implies preprocess is expected to be a string key.

    def test_calibrate(self):
        # More detailed test for calibrate
        conformalizer = ClassificationConformalizer(method='lac', preprocess='softmax', device='cpu')

        n_cal_samples = 10 # Increased from 3
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
        self.assertIsNotNone(conformalizer._score_function) # Should be initialized

    def test_calibrate_custom_score_fn(self):
        mock_score_fn = MagicMock(spec=ClassifNCScore)
        # Mock the behavior of the score function if needed for specific score values
        # For now, just ensure it's called
        mock_score_fn.return_value = torch.tensor(0.5) # Each call to score_fn returns a score

        conformalizer = ClassificationConformalizer(method=mock_score_fn, preprocess='softmax', device='cpu')

        n_cal_samples = 10 # Increased from 2
        n_classes = 3
        mock_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        alpha = 0.1

        conformalizer.calibrate(mock_preds, alpha, verbose=False)

        self.assertIs(conformalizer._score_function, mock_score_fn)
        # Check if the custom score function was called for each calibration sample
        self.assertEqual(mock_score_fn.call_count, len(mock_preds.true_cls))


    def test_conformalize_not_calibrated(self):
        conformalizer = ClassificationConformalizer(method='lac', preprocess='softmax')
        mock_eval_preds = MockClassificationPredictions(
            pred_cls=[torch.tensor([0.2, 0.5, 0.3])], true_cls=None, n_classes=3 # true_cls not needed for conformalize
        )
        with self.assertRaisesRegex(ValueError, "Conformalizer must be calibrated before conformalizing."):
            conformalizer.conformalize(mock_eval_preds)

    def test_conformalize(self):
        conformalizer = ClassificationConformalizer(method='lac', preprocess='softmax', device='cpu')

        # Calibration data
        n_cal_samples = 10 # Increased from 3
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
        mock_set_output = torch.tensor([0, 1]) # Example output for a prediction set
        conformalizer._score_function = MagicMock(spec=ClassifNCScore) # Replace real score_fn with mock
        conformalizer._score_function.get_set.return_value = mock_set_output

        conformalized_sets = conformalizer.conformalize(mock_eval_preds)

        self.assertIsInstance(conformalized_sets, list)
        self.assertEqual(len(conformalized_sets), len(eval_preds_list))
        self.assertTrue(conformalizer._score_function.get_set.called)
        self.assertEqual(conformalizer._score_function.get_set.call_count, len(eval_preds_list))

        for item in conformalized_sets:
            torch.testing.assert_close(item, mock_set_output)


    def test_evaluate_not_calibrated(self):
        conformalizer = ClassificationConformalizer(method='lac', preprocess='softmax')
        mock_preds = MockClassificationPredictions(pred_cls=[], true_cls=[], n_classes=3)
        # Conformalized predictions (dummy)
        conf_cls = [torch.tensor([0,1])]

        with self.assertRaisesRegex(ValueError, "Conformalizer must be calibrated before evaluating."):
            conformalizer.evaluate(mock_preds, conf_cls)

    def test_evaluate_conf_cls_none(self):
        conformalizer = ClassificationConformalizer(method='lac', preprocess='softmax')
         # Calibrate first
        n_cal_samples = 10 # Increased from 1
        n_classes = 3
        cal_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        conformalizer.calibrate(cal_preds, alpha=0.1, verbose=False)

        mock_preds = self._create_mock_preds(num_samples=0, n_classes=n_classes) # Empty eval data
        with self.assertRaisesRegex(ValueError, "Predictions must be conformalized before evaluating."):
            conformalizer.evaluate(mock_preds, None)


    def test_evaluate(self):
        conformalizer = ClassificationConformalizer(method='lac', preprocess='softmax', device='cpu')

        # Calibration
        n_cal_samples = 10 # Increased from 3
        n_classes = 3
        cal_preds = self._create_mock_preds(num_samples=n_cal_samples, n_classes=n_classes)
        alpha = 0.1
        conformalizer.calibrate(cal_preds, alpha, verbose=False)

        # Data for evaluation
        n_eval_samples = 3
        # Create true classes for evaluation
        eval_true_cls = [torch.randint(0, n_classes, (1,)).squeeze() for _ in range(n_eval_samples)]
        mock_eval_preds = MockClassificationPredictions(
            pred_cls=None, # Not strictly needed by evaluate if conf_cls is provided
            true_cls=eval_true_cls,
            n_classes=n_classes
        )

        # Sample conformalized prediction sets (list of tensors)
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
                if np.random.rand() > 0.5: # Randomly make sets larger
                     current_set.append(other_class)

            # Ensure all elements in current_set are valid class indices
            current_set = [cs % n_classes for cs in current_set]
            # Remove duplicates if any, though unlikely with this construction
            current_set = sorted(list(set(current_set)))

            conformalized_sets.append(torch.tensor(current_set))

            # For calculating expected coverage and set size
            if eval_true_cls[i].item() in current_set:
                expected_loss_sum += 1
            expected_set_size_sum += len(current_set)

        losses, set_sizes = conformalizer.evaluate(mock_eval_preds, conformalized_sets, verbose=False)

        self.assertIsInstance(losses, torch.Tensor)
        self.assertIsInstance(set_sizes, torch.Tensor)
        self.assertEqual(losses.shape[0], n_eval_samples)
        self.assertEqual(set_sizes.shape[0], n_eval_samples)

        expected_coverage = torch.tensor(expected_loss_sum / n_eval_samples if n_eval_samples > 0 else 0.0)
        expected_avg_set_size = torch.tensor(expected_set_size_sum / n_eval_samples if n_eval_samples > 0 else 0.0)

        torch.testing.assert_close(torch.mean(losses.float()), expected_coverage)
        torch.testing.assert_close(torch.mean(set_sizes.float()), expected_avg_set_size)

        if n_eval_samples > 0 and n_classes > 0: # Ensure we can make a non-covered case
            # Test with a case where one is not covered (if possible)
            conformalized_sets_partial_coverage = list(conformalized_sets) # copy
            # Make the first true_class not covered, if there are sets and classes to manipulate
            if len(conformalized_sets_partial_coverage) > 0 and n_classes > 0 :
                first_true_cls = eval_true_cls[0].item()
                # Create a new set for the first item that definitely does not include the true class
                non_covering_set = [(first_true_cls + 1) % n_classes]
                if n_classes > 1: # Add another if possible, different from first_true_cls
                    non_covering_set.append((first_true_cls + 2) % n_classes)
                non_covering_set = sorted(list(set(non_covering_set)))
                conformalized_sets_partial_coverage[0] = torch.tensor(non_covering_set)

                # Recalculate expected_loss_sum_partial
                expected_loss_sum_partial = 0
                for i in range(n_eval_samples):
                    if eval_true_cls[i].item() in conformalized_sets_partial_coverage[i].tolist():
                        expected_loss_sum_partial +=1

                losses_partial, _ = conformalizer.evaluate(
                    mock_eval_preds, conformalized_sets_partial_coverage, verbose=False
                )
                expected_coverage_partial = torch.tensor(expected_loss_sum_partial / n_eval_samples)
                torch.testing.assert_close(torch.mean(losses_partial.float()), expected_coverage_partial)


if __name__ == '__main__':
    # This allows running the tests directly from the script
    # However, typically you'd use a test runner like `python -m unittest discover`
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
