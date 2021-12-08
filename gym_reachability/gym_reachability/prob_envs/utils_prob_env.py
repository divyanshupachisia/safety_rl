import numpy as np
import scipy.signal

'''
Generate a 10 by 10 grid with values between 0 and 1 at each index
Values represent probability of obstacle 
'''
def gen_grid(): 
    # just for testing, need to come up with many such environments (perhaps parameterized in some way)
    # local maps must be floats
    grid = np.array([
            [0.9, 0.9, 0.8, 0.6, 0.3, 0.0, 0.3, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.8, 0.6, 0.2, 0.0, 0.2, 0.3, 0.9, 0.95],
            [0.9, 0.9, 0.4, 0.1, 0.1, 0.1, 0.2, 0.3, 0.9, 0.95],
            [0.9, 0.6, 0.0, 0.1, 0.1, 0.1, 0.2, 0.5, 0.9, 0.95],
            [0.9, 0.3, 0.2, 0.6, 1, 0.8, 0.6, 0.7, 0.9, 0.95],
            [0.9, 0.3, 0.2, 0.6, 1, 0.8, 0.6, 0.8, 0.9, 0.95],
            [0.9, 0.6, 0.0, 0.2, 0.2, 0.1, 0.1, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.6, 0.9, 0.95],
            [0.9, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.6, 0.9, 0.95],
            [0.9, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.95]
            ])
    
    return grid

'''
Given a sensing radius, R, the current robot position, cur_pos, and a grid environment 
return the subset of the grid environment as a vector that falls within the sensing radius
cur_pos is a tuple (i,j)
'''
def get_local_map(R, grid, cur_pos):

    # get the position
    x_pos = int(round(cur_pos[0]))
    y_pos = int(round(cur_pos[1]))
    print("Current Position: {},{}".format(x_pos,y_pos))

    # define square you need to iterate over
    x_range_low = x_pos - R
    x_range_high = x_pos + R
    y_range_low = y_pos - R 
    y_range_high = y_pos + R
    # TODO need to change this so that R can be a float (range won't work in loops below)

    row, col = grid.shape # to check if we're out of bounds
    local_map = [] # append entries here to return

    for i in range(x_range_low, x_range_high+1): 
        for j in range(y_range_low, y_range_high+1): 
            # check radius (since we want a circle, not square) 
            dist = np.sqrt((i-x_pos)**2  + (j-y_pos)**2)
            # continue if distance is greater than R
            if dist > R:
                continue
            else: 
                # check if you're out of bounds and if so append 1 (obstacle)
                if i < 0 or i > row-1 or j < 0 or j > col-1: 
                    local_map.append(1)

                # append value of grid if you're not out of bounds and are within R
                else: 
                    local_map.append(grid[i][j])
    return local_map  

'''
Return a low dimensional representation of the local grid around the current position.
cur_pos is a tuple (i,j)
'''
def conv_grid(cur_pos,filter=None, R=2):
    default_filter = [[1,1,1],[1,1,1],[1,1,1]]
    # if None then set to default
    if filter is None:
        filter = default_filter
    grid = gen_grid()
    cur_x = int(round(cur_pos[0]))
    cur_y = int(round(cur_pos[1]))
    local_grid = grid[cur_x-R:cur_x+R,cur_y-R:cur_y+R]
    conv = scipy.signal.convolve(filter, local_grid, mode='valid')
    states = conv.flatten()
    return states

# for testing
state = conv_grid(cur_pos=(3.2,3.2))