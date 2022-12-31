import torch
from torchmetrics import Metric
from typeguard import typechecked

# Logging
from gaze_verification import logging_handler
logger = logging_handler.get_logger(__name__,
                                    log_to_file=False)


@typechecked
class EER(Metric):
    """
    Calculates the value of the Equal Error Rate
    Equal Error Rate (EER): is the point where FNMR(t) = FMR(t).
    In practice the score distribution are not continuous so and interval
    is returned instead. The EER value will be set as the midpoint of
    this interval.
    The interval will be defined as:
    [EERlow, EERhigh] = min(fnmr[t], fmr[t]), max(fnmr[t], fmr[t])
    where t = argmin(abs(fnmr - fmr))
    The EER value is computed as (EERlow + EERhigh) / 2
    Reference:a
    Maio, D., Maltoni, D., Cappelli, R., Wayman, J. L., & Jain, A. K. (2002).
    FVC2000: Fingerprint verification competition. IEEE Transactions on
    Pattern Analysis and Machine Intelligence, 24(3), 402-412.

    returns: (index for EERlow and EERhigh, EERlow, EERhigh, EER)
    """
    def __init__(self):
        super().__init__()
        # False Match Rates (FMR)
        self.add_state("fmr", default=torch.tensor(0), dist_reduce_fx="cat")
        # False Non-Match Rates (FNMR)
        self.add_state("fnmr", default=torch.tensor(0), dist_reduce_fx="cat")

    def update(self, fmr: torch.Tensor, fnmr: torch.Tensor):
        """
        Updates the state variables of metric for a batch.
        """
        assert fmr.shape == fnmr.shape
        self.fmr = fmr
        self.fnmr = fnmr

    def compute(self):
        """
        Compute the final metric value from state variables.
        """
        diff = self.fmr - self.fnmr
        t2 = torch.where(diff <= 0)[0]

        if len(t2) > 0:
            t2 = t2[0]
        else:
            logger.warning('It seems that the FMR and FNMR curves do not intersect each other')
            return 0, 1, 1, 1

        t1 = t2 - 1 if diff[t2] != 0 and t2 != 0 else t2

        if self.fmr[t1] + self.fnmr[t1] <= self.fmr[t2] + self.fnmr[t2]:
            return t1, self.fnmr[t1], self.fmr[t1], (self.fnmr[t1] + self.fmr[t1]) / 2
        else:
            return t2, self.fmr[t2], self.fnmr[t2], (self.fnmr[t2] + self.fmr[t2]) / 2