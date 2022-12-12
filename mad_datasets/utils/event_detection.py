"""General event detection helper."""

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
