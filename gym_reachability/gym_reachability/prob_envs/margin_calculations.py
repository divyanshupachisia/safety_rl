import math
from env_utils import calculate_margin_rect
import numpy as np
from utils_prob_env import gen_grid # TODO remove after testing harness finished
import matplotlib.pyplot as plt



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

    closest_x = int(round(og_x))
    closest_y = int(round(og_y))

    safety_margin = 0

    # Add the current grid value to safety margin
    safety_margin = safety_margin + threshold*local_map[closest_x, closest_y]

    square_count = 1

    for i in range(int(round(closest_x-cutoff_radius)), int(round(closest_x+cutoff_radius+1))):
        for j in range(int(round(closest_y-cutoff_radius)), int(round(closest_y+cutoff_radius+1))):

            grid_distance = math.sqrt(float(((closest_x-i)*(closest_x-i)) + ((closest_y-j)*(closest_y-j))))
            actual_distance = math.sqrt(float(((og_x-i)*(og_x-i)) + ((og_y-j)*(og_y-j))))

            if (closest_x != i or closest_y != j) and grid_distance <= cutoff_radius: # Exclude the original cell and all cells farther from the distance

                square_count += 1

                # Inside the boundary
                if i >= 0 and i < len(local_map) and j >= 0 and j < len(local_map[0]):
                    safety_margin = safety_margin + (beta*1/(actual_distance + (beta/threshold)))*local_map[i][j]

                # Outside the boundary or on the boundary line
                else:
                    safety_margin = safety_margin + (beta*1/(actual_distance + (beta/threshold)))*1

    adjusted_margin = scaling_factor*(safety_margin-threshold)

    return adjusted_margin

def print_safety_margin(local_map, divisions):
    """Prints the safety margin of the given local_map, where each cell is divided into divisions^2 components.

    Args:
        local_map (2D np.ndarray): the local map
        divisions (int): each grid cell will have a safety_margin sample taken for divisions^2 times

    Returns:
        Nothing. The function prints the graph using pyplot
    """

    score_map = []

    for i in np.arange(-0.49, len(local_map)-0.51, 1/divisions):

        vector_map = []
        for j in np.arange(-0.49, len(local_map[0])-0.51, 1/divisions):
            vector_map.append(-1*safety_margin([i, j], scaling_factor, beta, cutoff_radius, threshold, local_map))

        score_map.append(vector_map)

    plt.subplot(1, 2, 1)
    plt.imshow(score_map, interpolation='nearest', cmap="RdBu")
    plt.title('Safety Margin Value Heat Map')

    plt.subplot(1, 2, 2)
    plt.imshow(-1*local_map, interpolation='nearest', cmap="RdBu")
    plt.title('Safety Margin Value Heat Map')

    plt.tight_layout()
    plt.show()





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

## Testing harness TODO remove

# Integer states
scaling_factor = 1
beta = 1
cutoff_radius = 2
threshold = 2.1*math.pi # Given these parameters, the characteristic radius is 1
local_map = gen_grid()

print(local_map)

score_map = []
for i in range(len(local_map)):

    vector_map = []

    for j in range(len(local_map)):

        vector_map.append(safety_margin([i, j], scaling_factor, beta, cutoff_radius, threshold, local_map))

    score_map.append(vector_map)

print(np.around(np.array(score_map), 1))
print_safety_margin(local_map, 10)

# Try putting robot in a block of 1 with small sensing radius to see if threshold is below

# Non-integer states
s = [4, 6]
print("State: [" + str(s[0]) + ", " + str(s[1]) + "] Value: " +str(safety_margin(s, scaling_factor, beta, cutoff_radius, threshold, local_map)))

s = [4.05, 5.95]
print("State: [" + str(s[0]) + ", " + str(s[1]) + "] Value: " +str(safety_margin(s, scaling_factor, beta, cutoff_radius, threshold, local_map)))

for i in np.arange(4, 5, 0.05):

        s = [i, 5.9]

        print("State: [" + str(s[0]) + ", " + str(s[1]) + "] Value: " + str(
            safety_margin(s, scaling_factor, beta, cutoff_radius, threshold, local_map)))

