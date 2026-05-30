import numpy as np
from typing import Union
from numpy.typing import NDArray


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length,) + shape


def unsqueeze_observation(
    observation: Union[dict[str, NDArray], NDArray]
) -> Union[dict[str, NDArray], NDArray]:
    """
    Unsqueeze the observation to add a batch dimension.
    """
    if isinstance(observation, dict):
        return {k: v[np.newaxis, :] for k, v in observation.items()}
    else:
        return observation[np.newaxis, :]
    

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])