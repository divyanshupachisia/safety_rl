import math
from .env_utils import calculate_margin_rect
import numpy as np

def safety_margin(s, scaling_factor, beta, cutoff_radius, threshold, local_map):
    """Computes the safety margin .

    Args:
        s (np.ndarray): the state of the agent.
        scaling_factor (positive float): the scaling factor for the environment
        beta (positive float): the coefficient of the g function (derived from the characteristic radius)
        cutoff_radius (positive float): only cells closer than cutoff_radius are included in the calculation
        threshold (positive float): the threshold for the safety margin that delineates any return value > 0 as unsafe
        local_map (2D np array of floats in [0,1]): The risk map for this environment

        To create a "safety bubble" of radius R, in which there is 0 probability of an obstacle (and assuming that
        all the space from R to cutoff_radius is an obstacle), then threshold < 2*pi*beta(cutoff_radius-R)

    Returns:
        float: positive numbers indicate being inside the failure set (safety violation).
    """

    og_x, og_y = s

    safety_margin = 0

    # Find closest grid point

    min_distance = math.inf
    closest_cell = [og_x, og_y]
    for i in range(int(og_x), int(og_x)+2):
        for j in range(int(og_y), int(og_y)+2):

            distance = math.sqrt((og_x-i)*(og_x-i) + (og_y-j)(og_y-j))

            if distance < min_distance:
                distance = min_distance
                closest_cell = [i, j]


    closest_x = closest_cell[0]
    closest_y = closest_cell[1]

    # Add the current grid value to safety margin
    safety_margin = safety_margin + threshold*local_map[closest_x, closest_y]

    for i in range(closest_x-cutoff_radius, closest_x+cutoff_radius+1):
        for j in range(closest_y-cutoff_radius, closest_y+cutoff_radius+1):

            distance = float(math.sqrt((og_x-i)*(og_x-i) + (og_y-j)(og_y-j)))

            if closest_x != i and closest_y != j and distance <= cutoff_radius: # Exclude the original cell and all cells farther from the distance

                # Inside the boundary
                if i >= 0 and i < len(local_map) and j >= 0 and j < len(local_map[0]):
                    safety_margin = safety_margin + (beta*1/(distance + (beta/threshold)))*local_map[i][j]

                # Outside the boundary or on the boundary line
                else:
                    safety_margin = safety_margin + (beta*1/(distance + (beta/threshold)))*1

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