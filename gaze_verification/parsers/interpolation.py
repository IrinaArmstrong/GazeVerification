import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional

# Logging
from gaze_verification import logging_handler
logger = logging_handler.get_logger(__name__,
                                    log_to_file=False)


def interpolate_session(session: pd.DataFrame,
                        columns: List[str],
                        interpolation_kwargs: Dict[str, Any],
                        agg_column: Optional[str] = None,
                        beaten_ratio: float = 30,
                        min_length: int = 500) -> pd.DataFrame:
    """
    Clear missing data to make correct filtration.
    (Disable its effect on filtration).
    """
    logger.info(f"Interpolating erroneous observations in data...")
    init_size = session.shape
    logger.info(f"Session before interpolation shape: {init_size[0]}")

    session['bad_sample'] = session.apply(lambda row: 1 if any([row[col] < 0 for col in columns]) else 0, axis=1)
    logger.info(f"{session['bad_sample'].sum()} erroneous observations detected in data.")

    # Make non valid frame x any as Nans
    for col in columns:
        session.loc[session['bad_sample'] == 1, col] = np.nan

    # Inside each trial - fill with interpolate values with Pandas splines
    data_interpolated = []
    beaten_cnt = 0

    for group_name, trial_data in tqdm(session.groupby(by=['trialId'])):
        if 100 * (trial_data['bad_sample'].sum() / trial_data.shape[0]) >= beaten_ratio:
            logger.info(f"Broken trial with ratio of beaten rows > {beaten_ratio}%")
            beaten_cnt += 1
            continue
        if trial_data.shape[0] < min_length:
            print(f"Too small trial with length < {min_length}: {trial_data.shape[0]}")
            beaten_cnt += 1
            continue

        trial_data[x] = trial_data[x].interpolate(**interpolation_kwargs)
        trial_data[y] = trial_data[y].interpolate(**interpolation_kwargs)

        if (sum(trial_data[x].isna()) > 0) or (sum(trial_data[y].isna()) > 0):
            trial_data = trial_data.loc[~trial_data[x].isna()]
            trial_data = trial_data.loc[~trial_data[y].isna()]
        data_interpolated.append(trial_data.reset_index(drop=True))

    session = pd.concat(data_interpolated, axis=0)
    logger.info(f"Session after interpolation shape: {session.shape[0]}, diff: {init_size[0] - session.shape[0]}")
    del data_interpolated

    return session
