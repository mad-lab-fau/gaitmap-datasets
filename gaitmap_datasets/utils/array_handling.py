import numpy as np


def bool_array_to_start_end_array(bool_array: np.ndarray) -> np.ndarray:
    """Find regions in bool array and convert those to start-end indices.

    The end index is inclusiv!

    Parameters
    ----------
    bool_array : array with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    array of [start, end] indices with shape (n,2)

    Examples
    --------
    >>> example_array = np.array([0,0,1,1,0,0,1,1,1])
    >>> start_end_list = bool_array_to_start_end_array(example_array)
    >>> start_end_list
    array([[2, 4],
           [6, 9]])
    >>> example_array[start_end_list[0, 0]: start_end_list[0, 1]]
    array([1, 1])

    """
    # check if input is actually a boolean array
    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean array!")

    if len(bool_array) == 0:
        return np.array([])

    slices = np.ma.flatnotmasked_contiguous(np.ma.masked_equal(bool_array, 0))
    return np.array([[s.start, s.stop] for s in slices])
