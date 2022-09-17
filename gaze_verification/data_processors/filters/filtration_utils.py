from gaze_verification.abstract_enum import AbstractEnumeration


class DerivativeOrder(AbstractEnumeration):
    """
    Provides enumeration for orders of derivatives.
    """
    NONE = 0  # no derivative
    FIRST_ORDER = 1  # 1 - for first order derivative - velocity
    SECOND_ORDER = 2  # 2 - for second order derivative - acceleration


class DerivativeType(AbstractEnumeration):
    """
    Provides enumeration for types of derivatives.
    """
    NONE = 0  # no derivative
    COLUMN = 1  # columns - axis 1 of array
    ROW = 2  # rows - axis 2 of array
    BOTH = 3   # for both directions: columns and rows
