import json
import unittest
from pathlib import Path

import numpy as np

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_objects.label import Label, ClassificationLabel
from gaze_verification.data_objects.target import Target, ClassificationTarget


class TestSamples(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._output_path = self._current_dir / "saved_samples"  # / "saves_test_samples.json"

    def test_sample_creation(self):
        data = np.zeros((10, 4), dtype=np.float32)
        s0 = Sample(guid=0, seq_id=0, session_id=0, data=data)
        s1 = Sample(guid=1, seq_id=1, session_id=0, label=0, user_id=0, data=data, predicted_label=1)

        target0 = ClassificationTarget(id=0, name="Peter")
        target1 = ClassificationTarget(id=1, name="Maria")
        pred_label = ClassificationLabel(labels=[target0, target1],
                                         probabilities={"Peter": 0.25, "Maria": 0.75})
        s2 = Sample(guid=2, seq_id=1, session_id=0,
                    label=target1, user_id=target1.id,
                    data=data, predicted_label=pred_label)
        print(s2)

    def test_sample_save_and_load_json(self):
        data = np.zeros((10, 4), dtype=np.float32).tolist()
        target0 = ClassificationTarget(id=0, name="Peter")
        target1 = ClassificationTarget(id=1, name="Maria")
        pred_label = ClassificationLabel(labels=[target0, target1],
                                         # probabilities={"Peter": 0.25, "Maria": 0.75}
                                         )
        sample = Sample(guid=2, seq_id=1, session_id=0,
                    label=target1, user_id=target1.id,
                    data=data, predicted_label=pred_label)
        print(sample)

        samples = Samples([sample])
        # Save
        json_path = self._output_path / "saves_test_samples.json"
        samples.save_json(path=str(json_path), as_single_file=True)
        self.assertTrue(json_path.exists())
        print(f"Samples saves successfully.\n")
        # Load
        samples_loaded = Samples.load_json(str(json_path))
        self.assertEqual(len(samples), len(samples_loaded))
        print(f"Samples loaded successfully.")

    def test_sample_save_and_load_pickle(self):
        data = np.zeros((10, 4), dtype=np.float32).tolist()
        target0 = ClassificationTarget(id=0, name="Peter")
        target1 = ClassificationTarget(id=1, name="Maria")
        pred_label = ClassificationLabel(labels=[target0, target1],
                                         # probabilities={"Peter": 0.25, "Maria": 0.75}
                                         )
        sample = Sample(guid=2, seq_id=1, session_id=0,
                    label=target1, user_id=target1.id,
                    data=data, predicted_label=pred_label)
        print(sample)

        samples = Samples([sample])
        # Save
        pickle_path = self._output_path / "saves_test_samples.pickle"
        samples.save_pickle(path=str(pickle_path), as_single_file=True)
        self.assertTrue(pickle_path.exists())
        print(f"Samples saves successfully.\n")
        # Load
        samples_loaded = Samples.load_pickle(str(pickle_path))
        self.assertEqual(len(samples), len(samples_loaded))
        print(f"Samples loaded successfully.")


if __name__ == '__main__':
    unittest.main()
