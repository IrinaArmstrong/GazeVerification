import numpy as np
import pandas as pd

from gaze_verification.logging_handler import get_logger

_logger = get_logger(
    name=__name__,
    logging_level="INFO"
)


def reduce_mem_usage(df, verbose: bool = False):
    """
    This function is used to reduce memory of a pandas dataframe.
    The idea is cast the numeric type to another more memory-effective type.
    It iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Code was taken from: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        _logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        _logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        _logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) /
                                                   start_mem))

    return df
