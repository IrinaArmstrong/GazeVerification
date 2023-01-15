import unittest
from pathlib import Path
from gaze_verification.data_objects.sample import Samples, Sample
from gaze_verification.data_processors.velocity_estimators import SimpleDiffVelocityEstimator


class TestSavitzkyGolayVelocityEstimator(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._init_data_path = self._current_dir / "test_data_samples" / "filtering_samples.pickle"

    def test_compute_velocity_sample(self):
        init_sample = Samples.load_pickle(str(self._init_data_path))
        # assuming here is single Sample in Samples
        sample_data = init_sample[0].data
        n_elements, n_dims = sample_data.shape

        sampling_frequency = 1000
        estimator = SimpleDiffVelocityEstimator(sampling_frequency=sampling_frequency,
                                                verbose=True)
        velocity_sample = estimator.compute_velocity_sample(init_sample[0])
        velocity_sample.data_type = "velocity"
        self.assertTupleEqual(velocity_sample.data.shape, (n_elements, n_dims))

    def test_compute_velocity_samples(self):
        init_samples = Samples.load_pickle(str(self._init_data_path))
        # assuming here is single Sample in Samples
        sample_data = init_samples[0].data
        n_elements, n_dims = sample_data.shape

        window_size = 7
        order = 3
        estimator = SavitzkyGolayVelocityEstimator(window_size, order,
                                                   rate=1,
                                                   keep_erroneous_samples=False, verbose=True)
        velocity_samples = estimator.compute_velocity_samples(init_samples)
        for sample in velocity_samples:
            sample.data_type = "velocity"
        self.assertTupleEqual(velocity_samples[0].data.shape, (n_elements, n_dims))


if __name__ == '__main__':
    unittest.main()
