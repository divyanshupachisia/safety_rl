import numpy as np

'''
Generate a 10 by 10 grid with values between 0 and 1 at each index
Values represent probability of obstacle 
'''
def gen_grid(): 
    # just for testing, need to come up with many such environments (perhaps parameterized in some way)
    grid = np.array([
            [0.9, 0.9, 0.8, 0.6, 0.3, 0.0, 0.6, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.8, 0.6, 0.3, 0.0, 0.6, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.4, 0.1, 0.1, 0.1, 0.3, 0.8, 0.9, 0.95], 
            [0.9, 0.9, 0.0, 0.1, 0.1, 0.1, 0.3, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.2, 0.6, 0.8, 0.8, 0.6, 0.8, 0.9, 0.95], 
            [0.9, 0.9, 0.2, 0.6, 0.8, 0.8, 0.6, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.0, 0.2, 0.2, 0.1, 0.1, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.95],
            [0.9, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.95], 
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

# for testing
grid = gen_grid()
local_map = get_local_map(2, grid, (0.3,0.6))