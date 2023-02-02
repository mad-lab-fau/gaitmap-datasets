"""General event detection helper."""
from typing import Dict, Literal, Tuple, Union

import numpy as np
import pandas as pd
from numpy.linalg import norm


def detect_min_vel(gyr: np.ndarray, min_vel_search_win_size: int) -> float:
    """Detect min vel within a given gyr sequence."""
    energy = norm(gyr, axis=-1) ** 2
    if min_vel_search_win_size >= len(energy):
        raise ValueError("The value chosen for min_vel_search_win_size_ms is too large. Should be around 100 ms.")
    energy_view = sliding_window_view(
        energy,
        window_length=min_vel_search_win_size,
        overlap=min_vel_search_win_size - 1,
    )
    # find window with lowest summed energy
    min_vel_start = np.argmin(np.sum(energy_view, axis=1))
    # min_vel event = middle of this window
    min_vel_center = min_vel_start + min_vel_search_win_size // 2
    return min_vel_center


def sliding_window_view(arr: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
       This function will return by default a view onto your input array, modifying values in your result will directly
       affect your input data which might lead to unexpected behaviour! If padding is disabled (default) last window
       fraction of input may not be returned! However, if `nan_padding` is enabled, this will always return a copy
       instead of a view of your input data, independent if padding was actually performed or not!

    Parameters
    ----------
    arr : array with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.
    window_length : int
        length of desired window (must be smaller than array length n)
    overlap : int
        length of desired overlap (must be smaller than window_length)
    nan_padding: bool
        select if last window should be nan-padded or discarded if it not fits with input array length. If nan-padding
        is enabled the return array will always be a copy of the input array independent if padding was actually
        performed or not!

    Returns
    -------
    windowed view (or copy for nan_padding) of input array as specified, last window might be nan padded if necessary to
    match window size

    Examples
    --------
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(arr = data, window_length = 5, overlap = 3, nan_padding = True)
    >>> windowed_view
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.,  8.],
           [ 6.,  7.,  8.,  9., nan]])

    """
    if overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # calculate length of necessary np.nan-padding to make sure windows and overlaps exactly fits data length
    n_windows = np.ceil((len(arr) - window_length) / (window_length - overlap)).astype(int)
    pad_length = window_length + n_windows * (window_length - overlap) - len(arr)

    # had to handle 1D arrays separately
    if arr.ndim == 1:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), (0, pad_length), constant_values=np.nan)

        new_shape = (arr.size - window_length + 1, window_length)
    else:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), [(0, pad_length), (0, 0)], constant_values=np.nan)

        shape = (window_length, arr.shape[-1])
        n = np.array(arr.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((arr.strides, arr.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)[0 :: (window_length - overlap)]

    view = np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.

    return view


def convert_sampling_rates_event_list(
    event_list: pd.DataFrame, old_sampling_rate: float, new_sampling_rate: float
) -> pd.DataFrame:
    """Convert sampling rate of a given data frame.

    Parameters
    ----------
    event_list : pd.DataFrame
        Data frame with data to be converted.
    old_sampling_rate : float
        Sampling rate of the data frame.
    new_sampling_rate : float
        Sampling rate to convert to.

    Returns
    -------
    pd.DataFrame
        Data frame with converted sampling rate.

    """
    if old_sampling_rate == new_sampling_rate:
        return event_list

    # For all columns we simply convert to the new sampling rate and round to the nearest integer
    return (event_list / old_sampling_rate * new_sampling_rate).round().astype(int)


def convert_segmented_stride_list(
    stride_list: Union[pd.DataFrame, Dict[str, pd.DataFrame]], target_stride_type: Literal["min_vel", "ic"]
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Convert a segmented stride list with detected events into other types of stride lists.

    During the conversion some strides might be removed.

    Parameters
    ----------
    stride_list
        Stride list to be converted
    target_stride_type
        The stride list type that should be converted to

    Returns
    -------
    converted_stride_list
        Stride list in the new format

    """
    stride_list_type = "single" if isinstance(stride_list, pd.DataFrame) else "multi"
    if stride_list_type == "single":
        return _segmented_stride_list_to_min_vel_single_sensor(stride_list, target_stride_type=target_stride_type)[0]
    return {
        k: _segmented_stride_list_to_min_vel_single_sensor(v, target_stride_type=target_stride_type)[0]
        for k, v in stride_list.items()
    }


def _segmented_stride_list_to_min_vel_single_sensor(
    stride_list: pd.DataFrame, target_stride_type: Literal["min_vel", "ic"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a segmented stride list with detected events into other types of stride lists.

    During the conversion some strides might be removed.
    Note, this function does not check if the input is a proper stride list.

    Parameters
    ----------
    stride_list
        Stride list to be converted
    target_stride_type
        The stride list type that should be converted to

    Returns
    -------
    converted_stride_list
        Stride list in the new format
    removed_strides
        Strides that were removed during the conversion.
        This stride list is still in the input format.

    """
    converted_stride_list = stride_list.copy()
    converted_stride_list["old_start"] = converted_stride_list["start"]
    converted_stride_list["old_end"] = converted_stride_list["end"]

    # start of each stride is now the new start event
    converted_stride_list["start"] = converted_stride_list[target_stride_type]
    # end of each stride is now the start event of the next strides
    # Breaks in the stride list will be filtered later
    converted_stride_list["end"] = converted_stride_list[target_stride_type].shift(-1)
    if target_stride_type == "min_vel":
        if "ic" in converted_stride_list.columns:
            # pre-ic of each stride is the ic in the current segmented stride
            converted_stride_list["pre_ic"] = converted_stride_list["ic"]
            # ic of each stride is the ic in the subsequent segmented stride
            converted_stride_list["ic"] = converted_stride_list["ic"].shift(-1)
        if "tc" in converted_stride_list.columns:
            # tc of each stride is the tc in the subsequent segmented stride
            converted_stride_list["tc"] = converted_stride_list["tc"].shift(-1)

    elif target_stride_type == "ic" and "tc" in converted_stride_list.columns:
        # As the ic occurs after the tc in the segmented stride, new tc is the tc of the next stride
        converted_stride_list["tc"] = converted_stride_list["tc"].shift(-1)

    # Find breaks in the stride list, which indicate the ends of individual gait sequences.
    breaks = (converted_stride_list["old_end"] - converted_stride_list["old_start"].shift(-1)).fillna(0) != 0

    # drop unneeded tmp columns
    converted_stride_list = converted_stride_list.drop(["old_start", "old_end"], axis=1)

    # Remove the last stride of each gait sequence as its end value is already part of the next gait sequence
    converted_stride_list = converted_stride_list[~breaks]

    # drop remaining nans (last list elements will get some nans by shift(-1) operation above)
    converted_stride_list = converted_stride_list.dropna(how="any")

    return converted_stride_list, stride_list[~stride_list.index.isin(converted_stride_list.index)]
