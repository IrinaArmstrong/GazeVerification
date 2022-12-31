import json
import unittest
from pathlib import Path
from gaze_verification.data_objects.sample import Samples
from gaze_verification.target_configurators.target_configurator import TargetConfigurator


class TestTargetConfigurator(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._config_path = self._current_dir / "targets_config_test" / "targets_config.json"
        self._init_data_path = self._current_dir / "test_data_samples" / "samples_test.pickle"

    def test_configurator_initialization(self):
        configurator = TargetConfigurator(str(self._config_path))

    def test_configurator_target_mappings(self):
        """
        Testing idx2target and target2idx mappings.
        """
        with open(str(self._config_path), 'r') as f:
            reference_config = json.load(f)
        target_to_skip = "8"
        ref_idx2target = [target_name for target_name in reference_config.keys() if target_name != target_to_skip]
        ref_target2idx = {target_name: int(i) for i, target_name in enumerate(reference_config.keys()) if
                          target_name != target_to_skip}

        configurator = TargetConfigurator(str(self._config_path))
        config_idx2targets = configurator.idx2target
        config_target2idx = configurator.target2idx
        self.assertListEqual(ref_idx2target, config_idx2targets)
        self.assertDictEqual(ref_target2idx, config_target2idx)

    def test_configurator_samples_splitting(self):
        """
        Testing samples splitting.
        """
        init_samples = Samples.load_pickle(str(self._init_data_path))
        configurator = TargetConfigurator(str(self._config_path))

        # split dataset samples based on targets split
        dataset_samples_split = configurator.split_samples(init_samples)

        print(f"Samples splits:")
        for split_name, split_samples in dataset_samples_split.items():
            print(f"{split_name} -> {len(split_samples)} samples")



if __name__ == '__main__':
    unittest.main()
