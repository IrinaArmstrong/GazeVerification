import torch
from torchmetrics import Metric
from typeguard import typechecked

from gaze_verification.metrics.utils import torch_unique

# Logging
from gaze_verification import logging_handler
logger = logging_handler.get_logger(__name__,
                                    log_to_file=False)


@typechecked
class FMR_FNMR(Metric):
    """
    False Non-Match Rate (FNMR) and False Match Rate (FMR) - the basic measures of accuracy of a biometric system.

    FNMR refers to the probability that two biometrics samples from the same user
    will be falsely recognized as a non-match.
    For example, an FNMR of 10% indicates that 10 in 100 verification attempts by genuine users will be rejected.

    FMR on the other hand refers to the probability that two biometrics samples
    from a different user will be falsely recognized as a match.
    For example, an FMR of 10% indicates that 10 in 100 verification attempts by impostor users will be accepted.

    In verification, FNMR and FMR are also known as False Rejection Rate (FRR) and False Acceptance Rate (FAR)
    respectively.

    FRR is a measurement of how often the system rejects genuine users
    while FAR is a measurement of how often the system accepts impostor users.
    FRR is given by FRR = total false rejection / total true attempts.
    FAR is given by FAR = total false acceptance / total false attempts.

    Since the verification is based on the threshold as we previously mentioned,
    it is impossible to minimize both FRR and FAR.
    Increasing threshold will result in decreased FAR but increased FRR and vice versa.
    As an alternative one can use the true acceptance rate (TAR) given by TAR = 1 — FRR.
    """
    def __init__(self):
        super().__init__()
        # Genuine matching scores
        self.add_state("gscores", default=torch.tensor(0), dist_reduce_fx="cat")
        # Impostor matching scores
        self.add_state("iscores", default=torch.tensor(0), dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Updates the state variables of metric for a batch.
        """
        assert predictions.shape == targets.shape
        gscores = torch.masked_select(predictions, targets > 0)
        iscores = torch.masked_select(predictions, targets == 0)

        # Add labels as second col
        self.gscores = torch.cat([gscores.unsqueeze(1),
                                  torch.ones_like(gscores).unsqueeze(1)], 1)
        self.iscores = torch.cat([iscores.unsqueeze(1),
                                  torch.zeros_like(iscores).unsqueeze(1)], 1)

    def compute(self):
        """
        Compute the final metric value from state variables.
        return: (thresholds, FMR, FNMR) or (thresholds, FM, FNM)
        """
        # Stacking scores
        scores = torch.cat([self.gscores, self.iscores], 0)

        #  Sorting & cumsum targets
        _, indices = torch.sort(scores[:, 0], 0)
        scores = scores[indices]
        cumul = torch.cumsum(scores[:, 1], dim=0)

        # Grouping scores
        thresholds, u_indices = torch_unique(scores[:, 0])

        # Calculating FNM and FM distributions
        fnm = cumul[u_indices] - scores[u_indices][:, 1]  # rejecting s < t
        fm = self.iscores.size(0) - (u_indices - fnm)

        fnm_rates = fnm / self.gscores.size(0)
        fm_rates = fm / self.iscores.size(0)
        return  thresholds, fm_rates, fnm_rates
