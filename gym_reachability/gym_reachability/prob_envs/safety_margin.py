import math
from .env_utils import calculate_margin_rect
import numpy as np

def safety_margin(scaling_factor, s, R, cutoff_radius, threshold, local_map):
    """Computes the safety margin .

    Args:
        scaling_factor (positive float): the scaling factor for the environment
        s (np.ndarray): the state of the agent.
        R (positive float): the characteristic radius of the g function
        cutoff_radius (positive float): only cells closer than cutoff_radius are included in the calculation
        threshold (positive float): the threshold for the safety margin that delineates any return value > 0 as unsafe
        local_map (2D np array of floats in [0,1]): The risk map for this environment

    Returns:
        float: positive numbers indicate being inside the failure set (safety violation).
    """

    x, y = s

    safety_margin = 0

    for i in range(x-cutoff_radius, x+cutoff_radius+1):
        for j in range(y-cutoff_radius, y+cutoff_radius+1):

            distance = math.sqrt((x-i)*(x-i) + (y-j)(y-j))

            if distance <= cutoff_radius:

                # Inside the boundary
                if i >= 0 and i < len(local_map) and j >= 0 and j < len(local_map[0]):
                    safety_margin = safety_margin + R*R/distance*(local_map[x][y])

                # Outside the boundary and on the boundary line
                else:
                    safety_margin = safety_margin + R*R/distance*1

    adjusted_margin = scaling_factor*(safety_margin-threshold)

    return adjusted_margin

def target_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: 0 or negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    l_x_list = []

    # target_set_safety_margin
    for _, target_set in enumerate(self.target_x_y_w_h):
      l_x = calculate_margin_rect(s, target_set, negativeInside=True)
      l_x_list.append(l_x)

    target_margin = np.max(np.array(l_x_list)) # TODO change this to the inverse formula?
                                               # TODO, if so, ensure that the value entered into the inverse function is > -1/alpha or equivalent

    return self.scaling * target_margin