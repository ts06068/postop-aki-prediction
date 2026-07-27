import unittest

import numpy as np
import pandas as pd

from postop_aki.evaluation import (
    bootstrap_confidence_intervals,
    calibration_table,
    decision_curve_net_benefit,
    discrimination_metrics,
    evaluate_reduced_model_subgroups,
    expected_calibration_error,
    predicted_risk_deciles,
    threshold_metrics,
    youden_threshold,
)


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_have_perfect_performance(self) -> None:
        # The expected Youden cutoff is 0.8; all metrics are 1.0.
        outcome = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.2, 0.8, 0.9])

        self.assertAlmostEqual(youden_threshold(outcome, probability), 0.8)
        self.assertEqual(
            set(threshold_metrics(outcome, probability, 0.5).values()), {1.0}
        )
        self.assertEqual(
            discrimination_metrics(outcome, probability),
            {"auroc": 1.0, "auprc": 1.0},
        )

    def test_bootstrap_is_reproducible_and_keeps_both_classes(self) -> None:
        outcome = np.array([0] * 9 + [1])
        probability = np.linspace(0.05, 0.95, 10)

        first = bootstrap_confidence_intervals(outcome, probability, n_bootstrap=25)
        second = bootstrap_confidence_intervals(outcome, probability, n_bootstrap=25)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.loc["auroc"] == 1.0).all())

    def test_quantile_calibration_table_and_ece(self) -> None:
        outcome = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.2, 0.8, 0.9])

        table = calibration_table(outcome, probability, n_bins=2)

        self.assertEqual(table["n"].tolist(), [2, 2])
        self.assertEqual(table["observed_event_rate"].tolist(), [0.0, 1.0])
        self.assertAlmostEqual(
            expected_calibration_error(outcome, probability, n_bins=2), 0.15
        )

    def test_decision_curve_and_risk_deciles(self) -> None:
        outcome = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.2, 0.8, 0.9])

        curve = decision_curve_net_benefit(outcome, probability, [0.5])
        deciles = predicted_risk_deciles(outcome, probability, n_deciles=2)

        self.assertAlmostEqual(curve.loc[0, "model_net_benefit"], 0.5)
        self.assertAlmostEqual(curve.loc[0, "treat_all_net_benefit"], 0.0)
        self.assertEqual(deciles["min_predicted_risk"].tolist(), [0.1, 0.8])
        self.assertEqual(deciles["max_predicted_risk"].tolist(), [0.2, 0.9])
        self.assertEqual(deciles["observed_event_rate"].tolist(), [0.0, 1.0])

    def test_reduced_model_subgroups_use_fixed_predictions(self) -> None:
        outcome = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        probability = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        subgroups = pd.DataFrame(
            {
                "sex": ["F", "F", "F", "F", "M", "M", "M", "M"],
                "cardiac_surgery": [0, 0, 1, 1, 0, 0, 1, 1],
            }
        )

        results = evaluate_reduced_model_subgroups(
            outcome,
            probability,
            subgroups,
            threshold=0.5,
            n_bootstrap=10,
        )

        self.assertEqual(len(results), 4)
        self.assertEqual(results["n"].tolist(), [4, 4, 4, 4])
        self.assertTrue((results["auroc"] == 1.0).all())
        self.assertTrue((results["accuracy"] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
