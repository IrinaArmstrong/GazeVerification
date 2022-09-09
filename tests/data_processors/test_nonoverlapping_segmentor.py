import json
import unittest
from pathlib import Path
from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.segmentors.nonoverlapping_segmentor import NonOverlappingSegmentor


class TestNonOverlappingSegmentor(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._init_data_path = self._current_dir / "test_data_samples" / "samples.pickle"

    def test_segmentation(self):
        init_sample = Samples.load_pickle(str(self._init_data_path))
        # assuming here is single Sample in Samples
        sample_data = init_sample[0].data
        n_elements, n_dims = sample_data.shape
        min_completness_ratio = 0.5

        for segment_length in range(2, n_elements):
            # expected
            n_complete_segments = n_elements // segment_length
            completness_ratio = (n_elements - n_complete_segments * segment_length) / segment_length
            n_incomplete_segments = 1 if completness_ratio >= min_completness_ratio else 0
            expected_segment_size = (n_dims, segment_length)

            segmentor = NonOverlappingSegmentor(segment_length=segment_length,
                                                min_completness_ratio=min_completness_ratio,
                                                fill_value=-100.0)
            segmented_sample = segmentor.run(init_sample)

            # resulted
            n_segments = len(segmented_sample)
            self.assertEqual(n_segments, n_complete_segments + n_incomplete_segments)
            for segment in segmented_sample:
                self.assertTupleEqual(segment.data.shape, expected_segment_size)