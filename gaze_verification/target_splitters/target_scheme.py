import enum
from gaze_verification.target_splitters.proportions_target_splitter import ProportionsTargetSplitter
from gaze_verification.target_splitters.timebased_target_splitter import TimebasedTargetSplitter


@enum.unique
class TargetScheme(enum.Enum):
    """
    Provides enumeration for specific dataset splitting schemas:
    - like in [1]: Divide the entire data set by participant as follows - 25% into the test sample,
        another 25% into the validation sample and the remaining 50% into the training sample.
        This does not take into account the time period of recording sessions.
    - like in [2]: Divide the whole dataset by participants in a similar way to the previous method.
        At the stage of splitting the test sample into template and authentication records,
        take into account the time period of recording sessions - authentication records
        are collected severely lagged behind the template records by some time delta T.

    [1] Makowski, S., Prasse, P., Reich, D.R., Krakowczyk, D., Jäger, L.A., & Scheffer, T. (2021).
        DeepEyedentificationLive: Oculomotoric Biometric Identification and Presentation-Attack Detection
        Using Deep Neural Networks. IEEE Transactions on Biometrics, Behavior, and Identity Science, 3, 506-518.
    [2] Lohr, D.J., & Komogortsev, O.V. (2022). Eye Know You Too: A DenseNet Architecture
        for End-to-end Eye Movement Biometrics.
    """
    PROPORTIONS_SPLIT = ProportionsTargetSplitter
    TIME_DEPENDED_SPLIT = TimebasedTargetSplitter

    def get_value(self):
        """
        Returns value for enumeration name.
        """
        return self.value

    @classmethod
    def get_available_names(cls):
        """
        Returns a list of available enumeration name.
        """
        return [member for member in cls.__members__.keys()]

    @classmethod
    def to_str(cls):
        s = " / ".join([member for member in cls.__members__.keys()])
        return s