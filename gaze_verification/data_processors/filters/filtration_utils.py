import enum

@enum.unique
class Derivative(enum.Enum):
    """
    Provides enumeration for types of derivatives.
    """
    NONE = 0  # no derivative
    FIRST_ORDER = 1  # 1 - for first order derivative - velocity
    SECOND_ORDER = 2  # 2 - for second order derivative - acceleration

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