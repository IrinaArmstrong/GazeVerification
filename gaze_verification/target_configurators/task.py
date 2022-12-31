import enum

@enum.unique
class Task(str, enum.Enum):
    """
    Provides enumeration for task types.
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

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