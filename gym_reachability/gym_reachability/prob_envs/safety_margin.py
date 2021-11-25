import math

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


def target_margin(scaling_factor, s, target_dims, R):
    """Computes the adjusted margin between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """

    t_x, t_y = [target_dims[0]+(target_dims[2]+1), target_dims[1]+(target_dims[2]+1)] # Obtain center coordinates from target_x_y_length (need to account for different square sizes)

    box4_target_margin = 0 # TODO finsish this

    target_margin = -R*R*1/(box4_target_margin + R) + R # Same function scale as the safety margin
    return scaling_factor * target_margin