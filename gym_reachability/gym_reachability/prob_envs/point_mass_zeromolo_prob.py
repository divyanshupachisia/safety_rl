"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements an environment considering the 2D point object dynamics.
This environemnt shows comparison between reach-avoid Q-learning and Sum
(Lagrange) Q-learning.
envType:
    'basic': corresponds to Fig. 1 and 2 in the paper.
    'show': corresponds to Fig. 3 in the paper.
"""

import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import random
from .env_utils import calculate_margin_rect
from .utils_prob_env import *
# from .margin_calculations import safety_margin
import math
# from .env_utils import calculate_margin_rect, gen_grid

class ProbZermeloShowEnv(gym.Env):

  def __init__(
      self, device, mode='RA', doneType='toEnd', thickness=.1,
      sample_inside_obs=False, envType='show',
  ):
    """Initializes the environment with given arguments.

    Args:
        device (str): device type (used in PyTorch).
        mode (str, optional): reinforcement learning type. Defaults to 'RA'.
        doneType (str, optional): conditions to raise `done flag in
            training. Defaults to 'toEnd'.
        thickness (float, optional): the thickness of the obstaclrs.
            Defaults to 0.1.
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        envType (str, optional): environment type. Defaults to 'show'.
    """
    self.envType = envType

    # State Bounds.
    if envType == 'basic' or envType == 'easy':
      self.bounds = np.array([[-0.5, 9.5], [-0.5, 9.5]])
    else:
      self.bounds = np.array([[-0.5, 9.5], [-0.5, 9.5]])
    self.low = self.bounds[:, 0]
    print("LOWER BOUND: {}".format(self.low))
    self.high = self.bounds[:, 1]
    print("UPPER BOUND: {}".format(self.high))
    self.sample_inside_obs = sample_inside_obs

    # Time-step Parameters.
    self.time_step = 0.05

    # Control Parameters.
    if envType == 'basic' or envType == 'easy':
      self.upward_speed = 2.
    else:
      self.upward_speed = .5
    self.horizontal_rate = 1.
    self.discrete_controls = np.array([[
        -self.horizontal_rate, self.upward_speed
    ], [0, self.upward_speed], [self.horizontal_rate, self.upward_speed]])

    # safety_margin parameters #
    self.beta = 1
    self.cutoff_radius = 2
    self.threshold = 1.2*math.pi
    # Given these parameters, the characteristic radius is 0.5

    # Local Map
    self.local_map = np.transpose(gen_grid())

    # Convolution filter
    self.conv_radius = 2
    self.filter = [[1,1,1],[1,1,1],[1,1,1]]
    self.stride = 1 # TODO put this in convolve function
    self.width_conv_filter_dimension = math.ceil(((2*self.conv_radius)-len(self.filter)+1)/self.stride) # TODO ask Divy to check this
    self.height_conv_filter_dimension = math.ceil(((2*self.conv_radius)-len(self.filter[0])+1)/self.stride) #TODO ask Divy to check this

    self.conv_states_count = self.width_conv_filter_dimension * self.height_conv_filter_dimension

    assert(len(conv_grid(cur_pos=[0, 0], filter=self.filter, R=self.conv_radius, stride=self.stride)) == self.conv_states_count) # TODO remove this

    # Constraint Set Parameters.
    # [X-position, Y-position, width, height]
    # if envType == 'basic':
    #   self.constraint_x_y_w_h = np.array([
    #       [1.25, 2, 1.5, 1.5],
    #       [-1.25, 2, 1.5, 1.5],
    #       [0, 6, 1.5, 1.5],
    #   ])
    #   self.constraint_type = ['R', 'L', 'C']
    # elif envType == 'easy':
    #   self.constraint_x_y_w_h = np.array([
    #       [1.25, 2, 1.5, 1.5],
    #       [-1.25, 2, 1.5, 1.5],
    #       [0, 6, 3., thickness],
    #   ])
    #   self.constraint_type = ['R', 'L', 'C']
    # else:
    #   self.constraint_x_y_w_h = np.array([
    #       [0., 1.5, 4., thickness],
    #       [0., 4., 4., thickness],
    #   ])
    #   self.constraint_type = ['C', 'C']

    # Target Set Parameters.
    if envType == 'basic' or envType == 'easy':
      self.target_x_y_w_h = np.array([[4.51, 7.51, 2, 2]])
    else:
      self.target_x_y_w_h = np.array([[4.51, 7.51, 2, 2]])

    # Gym variables.
    self.action_space = gym.spaces.Discrete(3)  # {left, up, right}
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(self.midpoint - self.interval / 2),
        np.float32(self.midpoint + self.interval / 2)
    )
    self.viewer = None

    # Set random seed.
    self.seed_val = 0
    self.set_seed(self.seed_val)

    # Cost Parameters
    self.penalty = 1.
    self.reward = -1.
    self.costType = 'sparse'
    self.scaling = 1.

    # mode: normal or extend (keep track of ell & g)
    self.mode = mode
    if mode == 'extend':
      self.state = np.zeros(3+self.conv_states_count)
    else:
      self.state = np.zeros(2+self.conv_states_count)
    self.doneType = doneType

    # Visualization Parameters
    # self.constraint_set_boundary = self.get_constraint_set_boundary()
    self.target_set_boundary = self.get_target_set_boundary()
    if envType == 'basic' or envType == 'easy':
      self.visual_initial_states = [
          np.append([4.0, 1.0], conv_grid(cur_pos=[4.0, 1.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
          np.append([4.0, 2.0], conv_grid(cur_pos=[4.0, 2.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
          np.append([3.0, 3.0], conv_grid(cur_pos=[3.0, 3.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
          np.append([2.0, 4.0], conv_grid(cur_pos=[2.0, 4.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
      ]

    else:
      self.visual_initial_states = [
        np.append([4.0, 1.0], conv_grid(cur_pos=[4.0, 1.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
        np.append([4.0, 2.0], conv_grid(cur_pos=[4.0, 2.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
        np.append([3.0, 3.0], conv_grid(cur_pos=[3.0, 3.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
        np.append([2.0, 4.0], conv_grid(cur_pos=[2.0, 4.0], filter=self.filter, R=self.conv_radius, stride=self.stride)),
      ]

    if mode == 'extend':
      self.visual_initial_states = \
          self.extend_state(self.visual_initial_states)

    print(
        "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
            self.mode, self.doneType, self.sample_inside_obs
        )
    )

    # for torch
    self.device = device

  def extend_state(self, states):
    """Extends the state to consist of max{ell, g}. Only used for mode='extend'.

    Args:
        states (np.ndarray): (x, y) position of states.

    Returns:
        np.ndarray: extended states.
    """
    new_states = []
    for state in states:
      l_x = self.target_margin(state)
      g_x = self.safety_margin(state)
      new_states.append(np.append(state, max(l_x, g_x)))
    return new_states

  def reset(self, start=None):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): state to reset the environment to.
            If None, pick the state uniformly at random. Defaults to None.

    Returns:
        np.ndarray: The state the environment has been reset to.
    """
    if start is None:
      self.state = self.sample_random_state(
          sample_inside_obs=self.sample_inside_obs
      )
    else:
      self.state = start
    return np.copy(self.state)

  def sample_random_state(self, sample_inside_obs=False):
    """Picks the state uniformly at random. NEW*** Includes the convolved grid state

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
        the obstacles if True. Defaults to False.

    Returns:
        np.ndarray: sampled initial state.
    """
    inside_obs = True
    # Repeat sampling until outside obstacle if needed.
    while inside_obs:
      xy_sample = np.random.uniform(low=self.low, high=self.high)
      new_state = np.append(xy_sample, conv_grid(cur_pos=xy_sample, filter=self.filter, R=self.conv_radius, stride=self.stride))
      g_x = self.safety_margin(xy_sample)
      inside_obs = (g_x > 0)
      if sample_inside_obs:
        break

    return new_state

  # == Dynamics ==
  def step(self, action):
    """Evolves the environment one step forward under given action.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        float: the standard cost used in reinforcement learning.
        bool: True if the episode is terminated.
        dict: consist of target margin and safety margin at the new state.
    """

    u = self.discrete_controls[action]
    state, [l_x, g_x] = self.integrate_forward(self.state, u)
    self.state = state

    fail = g_x > 0
    success = l_x <= 0

    # = `cost` signal
    if self.mode == 'RA':
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        cost = 0.
    else:
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        if self.costType == 'dense_ell':
          cost = l_x
        elif self.costType == 'dense':
          cost = l_x + g_x
        elif self.costType == 'sparse':
          cost = 0. * self.scaling
        elif self.costType == 'max_ell_g':
          cost = max(l_x, g_x)
        else:
          raise ValueError("invalid cost type!")

    # = `done` signal
    if self.doneType == 'toEnd':
      done = self.check_within_env(self.state)
    elif self.doneType == 'fail':
      done = fail
    elif self.doneType == 'TF':
      done = fail or success
    else:
      raise ValueError("invalid done type!")

    # = `info`
    if done and self.doneType == 'fail':
      info = {"g_x": self.penalty * self.scaling, "l_x": l_x}
    else:
      info = {"g_x": g_x, "l_x": l_x}
    return np.copy(self.state), cost, done, info

  def integrate_forward(self, state, u):
    """Integrates the dynamics forward by one step.

    Args:
        state (np.ndarray): x, y - position
                            [a, b, c, ....] flattened states from convolution
                            [z]  - optional, extra state dimension
                                capturing reach-avoid outcome so far)
        u (np.ndarray): contol inputs, consisting of v_x and v_y

    Returns:
        np.ndarray: next state.
    """
    if self.mode == 'extend':
      x, y = state[0:2]
      z = state[-1]
    else:
      x, y = state

    # one step forward
    x = x + self.time_step * u[0]
    y = y + self.time_step * u[1]

    l_x = self.target_margin(np.array([x, y]))
    g_x = self.safety_margin(np.array([x, y]))

    # Obtain next convolution

    conv_states = np.array(conv_grid(cur_pos=[x, y], filter=self.filter, R=self.conv_radius, stride=self.stride))

    state = np.append([x, y], conv_states)

    if self.mode == 'extend':
      z = min(z, max(l_x, g_x))
      state = np.append(state, [z])

    info = np.array([l_x, g_x])

    return state, info

  # == Setting Hyper-Parameters ==
  def set_costParam(
      self, penalty=1., reward=-1., costType='sparse', scaling=1.
  ):
    """
    Sets the hyper-parameters for the `cost` signal used in training, important
    for standard (Lagrange-type) reinforcement learning.

    Args:
        penalty (float, optional): cost when entering the obstacles or crossing
            the environment boundary. Defaults to 1.0.
        reward (float, optional): cost when reaching the targets.
            Defaults to -1.0.
        costType (str, optional): providing extra information when in
            neither the failure set nor the target set. Defaults to 'sparse'.
        scaling (float, optional): scaling factor of the cost. Defaults to 1.0.
    """
    self.penalty = penalty
    self.reward = reward
    self.costType = costType
    self.scaling = scaling

  def set_seed(self, seed):
    """Sets the seed for `numpy`, `random`, `PyTorch` packages.

    Args:
        seed (int): seed value.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    random.seed(self.seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  # def set_bounds(self, bounds):
  #   """Sets the boundary and the observation_space of the environment.
  #
  #   Args:
  #       bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
  #   """
  #   self.bounds = bounds
  #
  #   # Get lower and upper bounds
  #   self.low = np.array(self.bounds)[:, 0]
  #   self.high = np.array(self.bounds)[:, 1]
  #
  #   # Double the range in each state dimension for Gym interface.
  #   midpoint = (self.low + self.high) / 2.0
  #   interval = self.high - self.low
  #   self.observation_space = gym.spaces.Box(
  #       np.float32(midpoint - interval/2), np.float32(midpoint + interval/2)
  #   )

  def set_doneType(self, doneType):
    """Sets the condition to terminate the episode.

    Args:
        doneType (str): conditions to raise `done flag in training.
    """
    self.doneType = doneType

  def set_sample_type(self, sample_inside_obs=False, verbose=False):
    """Sets the type of the sampling method.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.sample_inside_obs = sample_inside_obs
    if verbose:
      print("sample_inside_obs-{}".format(self.sample_inside_obs))

  # == Getting Margin == 
  def safety_margin(self, s):
    """Computes the safety margin . Each grid cell is further divided into 25 parts for the purposes of the calculation

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
    divisions = 5
    
    # get all the parameters 
    scaling_factor = self.scaling
    beta = self.beta
    cutoff_radius = self.cutoff_radius
    threshold = self.threshold
    local_map = self.local_map

    # shape of grid
    rows,cols = local_map.shape

    # og_x, og_y = s
    og_x = s[0]
    og_y = s[1]

    closest_x = int(round(og_x))
    closest_y = int(round(og_y))

    # if rounding up makes it go out of bounds then round down 
    if closest_x > rows-1: 
        closest_x = rows-1
    if closest_y > cols-1: 
        closest_y = cols-1
    if closest_x < 0: 
        closest_x = 0
    if closest_y < 0: 
        closest_y = 0

    safety_margin = 0

    # Add the current grid value to safety margin
    safety_margin = safety_margin + threshold*local_map[closest_x, closest_y]/(divisions*divisions)

    square_count = 1

    for i in np.arange(og_x-cutoff_radius, og_x+cutoff_radius+(1/divisions), 1/divisions):
        for j in np.arange(og_y-cutoff_radius, og_y+cutoff_radius+(1/divisions), 1/divisions):

            grid_distance = math.sqrt(float(((closest_x-i)*(closest_x-i)) + ((closest_y-j)*(closest_y-j))))
            actual_distance = math.sqrt(float(((og_x-i)*(og_x-i)) + ((og_y-j)*(og_y-j))))

            closest_i = int(round(i))
            closest_j = int(round(j))

            if (closest_x != closest_i or closest_y != closest_j) and actual_distance <= cutoff_radius: # Exclude the original cell and all cells farther from the distance

                square_count += 1

                # Inside the boundary
                if closest_i >= 0 and closest_i < len(local_map) and closest_j >= 0 and closest_j < len(local_map[0]):
                    safety_margin = safety_margin + (beta*1/(actual_distance + (beta/threshold)))*local_map[closest_i][closest_j]/(divisions*divisions)

                # Outside the boundary or on the boundary line
                else:
                    safety_margin = safety_margin + (beta*1/(actual_distance + (beta/threshold)))*1/(divisions*divisions)

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

  # == Getting Information ==
  def check_within_env(self, state):
    """Checks if the robot is still in the environment.

    Args:
        state (np.ndarray): the state of the agent.

    Returns:
        bool: True if the agent is not in the environment.
    """
    outsideTop = (state[1] >= self.bounds[1, 1])
    outsideLeft = (state[0] <= self.bounds[0, 0])
    outsideRight = (state[0] >= self.bounds[0, 1])
    return outsideTop or outsideLeft or outsideRight

  def get_constraint_set_boundary(self):
    """Gets the constarint set boundary.

    Returns:
        np.ndarray: of the shape (#constraint, 5, 2). Since we use the box
            constraint in this environment, we need 5 points to plot the box.
            The last axis consists of the (x, y) position.
    """
    num_constarint_set = self.constraint_x_y_w_h.shape[0]
    constraint_set_boundary = np.zeros((num_constarint_set, 5, 2))

    for idx, constraint_set in enumerate(self.constraint_x_y_w_h):
      x, y, w, h = constraint_set
      x_l = x - w/2.0
      x_h = x + w/2.0
      y_l = y - h/2.0
      y_h = y + h/2.0
      constraint_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
      constraint_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

    return constraint_set_boundary

  def get_target_set_boundary(self):
    """Gets the target set boundary.

    Returns:
        np.ndarray: of the shape (#target, 5, 2). Since we use the box target
            in this environment, we need 5 points to plot the box. The last
            axis consists of the (x, y) position.
    """
    num_target_set = self.target_x_y_w_h.shape[0]
    target_set_boundary = np.zeros((num_target_set, 5, 2))

    for idx, target_set in enumerate(self.target_x_y_w_h):
      x, y, w, h = target_set
      x_l = x - w/2.0
      x_h = x + w/2.0
      y_l = y - h/2.0
      y_h = y + h/2.0
      target_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
      target_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

    return target_set_boundary

  def get_warmup_examples(self, num_warmup_samples=100):
    """Gets the warmup samples.

    Args:
        num_warmup_samples (int, optional): # warmup samples. Defaults to 100.

    Returns:
        np.ndarray: sampled states.
        np.ndarray: the heuristic values, here we used max{ell, g}.
    """
    x_min, x_max = self.bounds[0, :]
    y_min, y_max = self.bounds[1, :]

    xs = np.random.uniform(x_min, x_max, num_warmup_samples)
    ys = np.random.uniform(y_min, y_max, num_warmup_samples)
    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
    states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

    for i in range(num_warmup_samples):
      x, y = xs[i], ys[i]
      conv_states = conv_grid(cur_pos=[x, y], filter=self.filter, R=self.conv_radius, stride=self.stride)
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))
      heuristic_v[i, :] = np.maximum(l_x, g_x)
      states[i, :] = np.append([x, y], conv_states)

    return states, heuristic_v

  def get_axes(self):
    """Gets the axes bounds and aspect_ratio.

    Returns:
        np.ndarray: axes bounds.
        float: aspect ratio.
    """
    x_span = self.bounds[0, 1] - self.bounds[0, 0]
    y_span = self.bounds[1, 1] - self.bounds[1, 0]
    aspect_ratio = x_span / y_span
    axes = np.array([
        self.bounds[0, 0] - .05, self.bounds[0, 1] + .05,
        self.bounds[1, 0] - .05, self.bounds[1, 1] + .05
    ])
    return [axes, aspect_ratio]

  def get_value(self, q_func, nx=41, ny=121, addBias=False):
    """Gets the state values given the Q-network.

    Args:
        q_func (object): agent's Q-network.
        nx (int, optional): # points in x-axis. Defaults to 41.
        ny (int, optional): # points in y-axis. Defaults to 121.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.

    Returns:
        np.ndarray: x-position of states
        np.ndarray: y-position of states
        np.ndarray: values
    """
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=['multi_index'])
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    while not it.finished:
      idx = it.multi_index

      x = xs[idx[0]]
      y = ys[idx[1]]
      conv_states = conv_grid(cur_pos=[x, y], filter=self.filter, R=self.conv_radius, stride=self.stride)
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))

      if self.mode == 'normal' or self.mode == 'RA':
        state = torch.FloatTensor(np.append([x, y], conv_states)).to(self.device).unsqueeze(0)
      else:
        z = max([l_x, g_x])
        state = torch.FloatTensor(np.append([x, y], conv_states, [z])).to(self.device).unsqueeze(0)

      if addBias:
        v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
      else:
        v[idx] = q_func(state).min(dim=1)[0].item()
      it.iternext()
    return xs, ys, v

  # == Trajectory Functions ==
  def simulate_one_trajectory(
      self, q_func, T=250, state=None, keepOutOf=False, toEnd=False
  ):
    """Simulates the trajectory given the state or randomly initialized.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        state (np.ndarray, optional): if provided, set the initial state to its
            value. Defaults to None.
        keepOutOf (bool, optional): smaple states inside obstacles if False.
            Defaults to False.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary if True. Defaults to False.

    Returns:
        np.ndarray: x-positions of the trajectory.
        np.ndarray: y-positions of the trajectory.
        int: the binary reach-avoid outcome.
    """
    if state is None:
      state = self.sample_random_state(sample_inside_obs=not keepOutOf)
    x, y = state[:2]
    traj_x = [x]
    traj_y = [y]
    result = 0  # not finished

    for _ in range(T):
      if toEnd:
        outsideTop = (state[1] > self.bounds[1, 1])
        outsideLeft = (state[0] < self.bounds[0, 0])
        outsideRight = (state[0] > self.bounds[0, 1])
        done = outsideTop or outsideLeft or outsideRight
        if done:
          result = 1
          break
      else:
        if self.safety_margin(state[:2]) > 0:
          result = -1  # failed
          break
        elif self.target_margin(state[:2]) <= 0:
          result = 1  # succeeded
          break

      state_tensor = torch.FloatTensor(state)
      state_tensor = state_tensor.to(self.device).unsqueeze(0)
      action_index = q_func(state_tensor).min(dim=1)[1].item()
      u = self.discrete_controls[action_index]

      state, _ = self.integrate_forward(state, u)
      traj_x.append(state[0])
      traj_y.append(state[1])

    return traj_x, traj_y, result

  def simulate_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, toEnd=False
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarrays, optional): if provided, set the initial
            states to its value. Defaults to None.
        toEnd (bool, optional): simulate the trajectory until the robot crosses
            the boundary if True. Defaults to False.

    Returns:
        list of np.ndarrays: each element is a tuple consisting of x and y
            positions along the trajectory.
        np.ndarray: the binary reach-avoid outcomes.
    """

    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
      if self.envType == 'basic' or self.envType == 'easy':
        nx = 21
        ny = 61
      else:
        nx = 41
        ny = nx
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      results = np.empty((nx, ny), dtype=int)
      it = np.nditer(results, flags=['multi_index'])
      print()
      while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]
        conv_state = conv_grid(cur_pos=[x, y], filter=self.filter, R=self.conv_radius, stride=self.stride)
        state = np.append([x, y], conv_state)
        traj_x, traj_y, result = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj_x, traj_y))
        results[idx] = result
        it.iternext()
      results = results.reshape(-1)
    else:
      results = np.empty(shape=(len(states),), dtype=int)
      for idx, state in enumerate(states):
        traj_x, traj_y, result = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj_x, traj_y))
        results[idx] = result

    return trajectories, results

  # == Visualizing ==
  def render(self):
    pass

  def visualize(
      self, q_func, vmin=-1, vmax=1, nx=201, ny=201, labels=None,
      boolPlot=False, addBias=False, cmap='seismic'
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.

    Args:
        q_func (object): agent's Q-network.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        labels (list, optional): x- and y- labels. Defaults to None.
        boolPlot (bool, optional): plot the values in binary form if True.
            Defaults to False.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
        cmap (str, optional): color map. Defaults to 'seismic'.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    cbarPlot = True

    # == Plot failure / target set ==
    self.plot_target_failure_set(ax)

    # == Plot reach-avoid set ==
    # self.plot_reach_avoid_set(ax)

    # == Plot V ==
    self.plot_v_values(
        q_func, ax=ax, fig=fig, vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
        boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias
    )

    # == Plot Trajectories ==
    self.plot_trajectories(
        q_func, states=self.visual_initial_states, toEnd=False, ax=ax
    )

    # == Formatting ==
    self.plot_formatting(ax=ax, labels=labels)
    fig.tight_layout()

  def plot_v_values(
      self, q_func, ax=None, fig=None, vmin=-1, vmax=1, nx=201, ny=201,
      cmap='seismic', alpha=0.8, boolPlot=False, cbarPlot=True, addBias=False
  ):
    """Plots state values.

    Args:
        q_func (object): agent's Q-network.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        fig (matplotlib.figure, optional): Defaults to None.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        cmap (str, optional): color map. Defaults to 'seismic'.
        alpha (float, optional): opacity. Defaults to 0.8.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        cbarPlot (bool, optional): plot the color bar if True.
            Defaults to True.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
    """
    axStyle = self.get_axes()

    # == Plot V ==
    _, _, v = self.get_value(q_func, nx, ny, addBias=addBias)
    vmax = np.ceil(max(np.max(v), np.max(-v)))
    vmin = -vmax

    if boolPlot:
      im = ax.imshow(
          v.T > 0., interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, alpha=alpha
      )
    else:
      im = ax.imshow(
          v.T, interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
      )
      if cbarPlot:
        cbar = fig.colorbar(
            im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
            ticks=[vmin, 0, vmax]
        )
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)

  def plot_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, toEnd=False,
      ax=None, c='k', lw=2, zorder=2
  ):
    """Plots trajectories given the agent's Q-network.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarrays, optional): if provided, set the initial
            states to its value. Defaults to None.
        toEnd (bool, optional): simulate the trajectory until the robot crosses
            the boundary if True. Defaults to False.
        ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
        c (str, optional): color of the trajectories. Defaults to 'k'.
        lw (float, optional): linewidth of the trajectories. Defaults to 2.
        zorder (int, optional): graph layers order. Defaults to 2.
    Returns:
        np.ndarray: the binary reach-avoid outcomes.
    """

    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))

    trajectories, results = self.simulate_trajectories(
        q_func, T=T, num_rnd_traj=num_rnd_traj, states=states, toEnd=toEnd
    )

    for traj in trajectories:
      traj_x, traj_y = traj
      ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
      ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

    return results

  def plot_target_failure_set(
      self, ax=None, c_c='m', c_t='y', lw=1.5, zorder=1
  ):
    """Plots the target and the failure set.

    Args:
        ax (matplotlib.axes.Axes, optional)
        c_c (str, optional): color of the constraint set boundary.
            Defaults to 'm'.
        c_t (str, optional): color of the target set boundary.
            Defaults to 'y'.
        lw (float, optional): liewidth. Defaults to 1.5.
        zorder (int, optional): graph layers order. Defaults to 1.
    """

    # Plot boundaries of constraint set.
    # for one_boundary in self.constraint_set_boundary:
    #   ax.plot(
    #       one_boundary[:, 0], one_boundary[:, 1], color=c_c, lw=lw,
    #       zorder=zorder
    #   )
    #

    # Plot boundaries of target set.
    for one_boundary in self.target_set_boundary:
      ax.plot(
          one_boundary[:, 0], one_boundary[:, 1], color=c_t, lw=lw,
          zorder=zorder
      )

  def plot_reach_avoid_set(self, ax=None, c='g', lw=3, zorder=1):
    """Plots the analytic reach-avoid set.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
        c (str, optional): color of the rach-avoid set boundary.
            Defaults to 'g'.
        lw (int, optional): liewidth. Defaults to 3.
        zorder (int, optional): graph layers order. Defaults to 1.
    """
    slope = self.upward_speed / self.horizontal_rate

    def get_line(slope, end_point, x_limit, ns=100):
      x_end, y_end = end_point
      b = y_end - slope*x_end

      xs = np.linspace(x_limit, x_end, ns)
      ys = xs*slope + b
      return xs, ys

    # unsafe set
    for cons, cType in zip(self.constraint_x_y_w_h, self.constraint_type):
      x, y, w, h = cons
      x1 = x - w/2.0
      x2 = x + w/2.0
      y_min = y - h/2.0
      if cType == 'C':
        xs, ys = get_line(-slope, end_point=[x1, y_min], x_limit=x)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
        xs, ys = get_line(slope, end_point=[x2, y_min], x_limit=x)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
      elif cType == 'L':
        x_limit = self.bounds[0, 0]
        xs, ys = get_line(slope, end_point=[x2, y_min], x_limit=x_limit)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
      elif cType == 'R':
        x_limit = self.bounds[0, 1]
        xs, ys = get_line(-slope, end_point=[x1, y_min], x_limit=x_limit)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

    # border unsafe set
    x, y, w, h = self.target_x_y_w_h[0]
    x1 = x - w/2.0
    x2 = x + w/2.0
    y_max = y + h/2.0
    xs, ys = get_line(slope, end_point=[x1, y_max], x_limit=self.low[0])
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
    xs, ys = get_line(-slope, end_point=[x2, y_max], x_limit=self.high[0])
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

  def plot_formatting(self, ax=None, labels=None):
    """Formats the visualization.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
        labels (list, optional): x- and y- labels. Defaults to None.
    """
    axStyle = self.get_axes()
    # == Formatting ==
    ax.axis(axStyle[0])
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    ax.grid(False)
    if labels is not None:
      ax.set_xlabel(labels[0], fontsize=52)
      ax.set_ylabel(labels[1], fontsize=52)

    ax.tick_params(
        axis='both', which='both', bottom=False, top=False, left=False,
        right=False
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
