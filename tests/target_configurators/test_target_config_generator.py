import json
import unittest
from pathlib import Path
from gaze_verification.data_objects.sample import Samples
from gaze_verification.target_configurators.target_config_generator import TargetConfigGenerator


class TestTargetConfigGenerator(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._config_path = self._current_dir / "targets_config_test"
        self._init_data_path = self._current_dir / "test_data_samples" / "samples_test.pickle"

    def test_config_generator_initialization(self):
        generator = TargetConfigGenerator(output_dir=str(self._config_path),
                                          target_scheme="PROPORTIONS_SPLIT")

    def test_config_generator_generation(self):
        """
        Testing mappings creation & splitting.
        """
        init_samples = Samples.load_pickle(str(self._init_data_path))
        generator = TargetConfigGenerator(output_dir=self._config_path,
                                          target_scheme="PROPORTIONS_SPLIT")
        targets_split = generator.run(init_samples,
                                      splits_proportions=[("train", 0.6),
                                                          ("test", 0.2),
                                                          ("validation", 0.2)],
                                      seed=11,
                                      min_targets_per_split=2,
                                      target_field='label')

        print(f"\nFound {len(targets_split)} unique target classes")
