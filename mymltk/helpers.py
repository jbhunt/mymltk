import torch
import numpy as np

def to_tensor(x, dtype=torch.double):
    """
    """

    if type(x) == np.ndarray:
        x_out = torch.from_numpy(x).to(dtype=dtype)
    else:
        x_out = x

    return x_out

def rescale(x, output_x_min=0, output_x_max=1):
    """
    """

    input_x_min = x.min()
    input_x_range = x.max() - input_x_min
    output_x_range = output_x_max - output_x_min
    x_scaled = ((x - input_x_min) * output_x_range) / input_x_range + output_x_min

    return x_scaled