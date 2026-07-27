import unittest
from pathlib import Path

from postop_aki.config import load_experiment_config
from postop_aki.data import load_modeling_data, split_cohorts
from postop_aki.features import get_feature_names
from postop_aki.models import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectIntegrityTests(unittest.TestCase):
    def test_feature_counts_match_frozen_models(self) -> None:
        # Each outcome keeps the manuscript variants in full/preop/reduced order.
        expected_counts = {"full": 55, "preoperative": 37, "reduced": 8}
        for config_name in ("any_aki.yaml", "severe_aki.yaml"):
            config = load_experiment_config(PROJECT_ROOT / "configs" / config_name)
            for model in config.models:
                self.assertEqual(
                    len(get_feature_names(model.feature_set)),
                    expected_counts[model.name],
                )

    def test_frozen_model_checksums(self) -> None:
        for config_name in ("any_aki.yaml", "severe_aki.yaml"):
            config = load_experiment_config(PROJECT_ROOT / "configs" / config_name)
            for model in config.models:
                artifact = config.project_root / model.artifact
                self.assertEqual(file_sha256(artifact), model.sha256)

    def test_sample_contains_only_small_synthetic_cohorts(self) -> None:
        sample = load_modeling_data(PROJECT_ROOT / "data" / "sample.xlsx")
        cohorts = split_cohorts(sample)
        self.assertEqual(len(cohorts.internal), 5)
        self.assertEqual(len(cohorts.external), 5)
        self.assertTrue(sample["PID"].astype(str).str.startswith("SYNTH-").all())


if __name__ == "__main__":
    unittest.main()
