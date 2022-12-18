import unittest
from pathlib import Path
from gaze_verification.data_objects.sample import Samples
from gaze_verification.data_processors.datasets import SamplesDataset
from gaze_verification.data_processors.segmentors.nonoverlapping_segmentor import NonOverlappingSegmentor


class TestSamplesDataset(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._init_data_path = self._current_dir / "test_data_samples" / "segmentation_samples.pickle"

    def test_creation(self):
        init_samples = Samples.load_pickle(str(self._init_data_path))
        dataset = SamplesDataset(init_samples, lambda x: x)
        self.assertEqual(len(init_samples), len(dataset))

    def test_preparing_samples(self):
        kwargs = dict(segment_length=1000,
                      min_completness_ratio=0.85,
                      fill_value=-100)

        # Create `prepare_sample_fn`
        def prepare_samples(samples: Samples, segment_length: int,
                            min_completness_ratio: float, fill_value: float):
            segmentor = NonOverlappingSegmentor(
                segment_length=segment_length,
                min_completness_ratio=min_completness_ratio,
                fill_value=fill_value)
            segmented_samples = segmentor.run(samples)
            return segmented_samples

        init_samples = Samples.load_pickle(str(self._init_data_path))
        dataset = SamplesDataset(init_samples, prepare_samples, kwargs)
        # self.assertEqual(len(init_samples), len(dataset))


if __name__ == '__main__':
    unittest.main()
