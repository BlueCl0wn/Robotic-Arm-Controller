from argparse import Namespace
import datetime
import os
from tensorboardX import SummaryWriter
import config
import numpy as np


def splash_screen(params: Namespace) -> str:
    """
    Creates splash screen for training.
    :param params: Namespace containing info of training run
    :return: text for splash screen
    """

    run_name = config.RUN_NAME  # "_" + params.commit[:8] + "_" + params.version
    print(f" Training Network".center(88, "="))
    print(f" Run Name: {run_name} ".center(88, "="))
    print(f"   Running on device {params.device}   ".center(88, "="))

    train_summary_writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR_ROOT, run_name))

    nl = "\n"
    train_summary_writer.add_text("run_name", f"""
### {run_name}
#### {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
---
## params
{nl.join([f"- `{k}`: `{v}`" for k, v in params.__dict__.items()])}
        """, 0)

    train_summary_writer.flush()

    return train_summary_writer


def get_file_descriptor(params: Namespace, episode: int) -> str:
    """
    Creates filename for a training run.

    :param params: Namespace of training run
    :param episode: episode number at moment of saving
    :return: str
    """
    return f"{config.MODELS_DIR}/{config.RUN_NAME}_{episode}.pth" #  '_{params.commit[:8]}'


def flatten_dict(d: dict[np.ndarray]) -> np.ndarray:
    """
    Flatten a dictionary of n-dimensional NumPy ndarrays into a single n-dimensional NumPy ndarray.

    This function takes a dictionary where the values are NumPy ndarrays,
    flattens each array, and concatenates them into a single 1D ndarray.

    Parameters:
    -----------
    d : dict
        A dictionary where the values are NumPy ndarrays. The keys can be
        of any hashable type. The arrays can have any shape or dimension.

    Returns:
    --------
    numpy.ndarray
        A 1D NumPy ndarray containing all elements from the input arrays,
        concatenated in the order they appear in the dictionary.

    Examples:
    ---------
    >> import numpy as np
    >> d = {'a': np.array([[1, 2], [3, 4]]),
    ...      'b': np.array([5, 6, 7]),
    ...      'c': np.array([[8], [9], [10]])}
    >> flatten_dict_of_ndarrays(d)
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    Notes:
    ------
    - The order of elements in the output array depends on the order of
      items in the input dictionary and the order of elements in each array.
    - This function uses numpy.concatenate(), which is efficient for
      combining multiple arrays.
    """
    return np.concatenate([arr.flatten() for arr in d.values()])