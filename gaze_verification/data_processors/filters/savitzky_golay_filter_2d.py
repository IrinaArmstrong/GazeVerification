import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from typing import List, Union, Optional

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.filters.filter_abstract import FilterAbstract
from gaze_verification.data_processors.filters.filtration_utils import Derivative


@typechecked
class SavitzkyGolayFilter2D(FilterAbstract):
    """
    The Savitzky Golay filter is a particular type of low-pass filter, well adapted for data smoothing.

    The idea of Savitzky-Golay filters is simple – for each sample in the filtered sequence,
    take its direct neighborhood of N neighbors and fit a polynomial to it.
    Then just evaluate the polynomial at its center (and the center of the neighborhood), point 0,
    and continue with the next neighborhood.

    Also Savitzky-Golay filters producing smoothened signal gradients / derivatives.
    The main point of Savitzky-Golay gradients is to compute the derivative at a given point,
    simply by computing the derivative of the fitted polynomial.

    Those filters are useful for filtering out the noise when the frequency span of the signal is large.
    They are reported to be optimal for minimizing the least-square error
    in fitting a polynomial to frames of the noisy data.

    For further information see:
        description: https://bartwronski.com/2021/11/03/study-of-smoothing-filters-savitzky-golay-filters/
        implementation details: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    def __init__(self,
                 window_size: int,
                 order: int,
                 deriviate: Optional[Union[str, int, Derivative]] = Derivative.NONE,
                 verbose: bool = True):
        super().__init__(verbose)
        self.window_size = window_size
        self.order = order
        self.deriviate = deriviate

    def filter_sample(self, sample: Sample) -> Sample:
        """
        Filter data sequences from Samples according to predefined logic.

        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :return: Sample object with filtered data field,
        :rtype: Sample
        """
