"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 2-dimensional point mass
problem. We use this script to generate Fig. 2 and Fig. 3 in the paper.

Examples:
    RA:
        python3 sim_naive.py -w -sf -of scratch -a -g 0.99 -n anneal
        python3 sim_naive.py -w -sf -of scratch -n 9999
        python3 sim_naive.py -w -sf -of scratch -g 0.999 -dt fail -n 999
    Lagrange:
        python3 sim_naive.py -sf -m lagrange -of scratch -g 0.95 -n 95
        python3 sim_naive.py -sf -m lagrange -of scratch -dt TF -g 0.95 -n 95
    test: python3 sim_naive.py -w -sf -of scratch -wi 100 -mu 100 -cp 40
"""

import os
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from RARL.DDQNSingle import DDQNSingle
from RARL.config import dqnConfig
from RARL.utils import save_obj
from gym_reachability.gym_reachability.prob_envs.utils_prob_env import *

# TODO remove (for Jacob's computing cluster)
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(torch.cuda.device_count()/2))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
except:
    print("No CUDA detected")

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

# == ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toEnd',
    type=str
)
parser.add_argument(
    "-ct", "--costType", help="cost type", default='sparse', type=str
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-r", "--reward", help="when entering target set", default=-1, type=float
)
parser.add_argument(
    "-p", "--penalty", help="when entering failure set", default=1, type=float
)
parser.add_argument(
    "-s", "--scaling", help="scaling of ell/g", default=4, type=float
)

# training scheme
parser.add_argument(
    "-w", "--warmup", help="warmup Q-network", action="store_true"
)
parser.add_argument(
    "-wi", "--warmupIter", help="warmup iteration", default=2000, type=int
)
parser.add_argument(
    "-mu", "--maxUpdates", help="maximal #gradient updates", default=1200000, # TODO change this back to 400000
    type=int
)
parser.add_argument(
    "-ut", "--updateTimes", help="#hyper-param. steps", default=10, type=int
)
parser.add_argument(
    "-mc", "--memoryCapacity", help="memoryCapacity", default=10000, type=int
)
parser.add_argument(
    "-cp", "--checkPeriod", help="check period", default=20000, type=int
)

# NN hyper-parameters
parser.add_argument(
    "-a", "--annealing", help="gamma annealing", action="store_true"
)
parser.add_argument(
    "-arc", "--architecture", help="NN architecture", default=[200, 40, 20], # TODO change back to [100, 20]
    nargs="*", type=int
)
parser.add_argument(
    "-lr", "--learningRate", help="learning rate", default=1e-3, type=float
)
parser.add_argument(
    "-g", "--gamma", help="contraction coeff.", default=0.9999, type=float
)
parser.add_argument(
    "-act", "--actType", help="activation type", default='Tanh', type=str
)

# RL type
parser.add_argument("-m", "--mode", help="mode", default='RA', type=str)
parser.add_argument(
    "-tt", "--terminalType", help="terminal value", default='g', type=str
)

# file
parser.add_argument(
    "-st", "--showTime", help="show timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default='', type=str)
parser.add_argument(
    "-of", "--outFolder", help="output file", default='experiments', type=str
)
parser.add_argument(
    "-pf", "--plotFigure", help="plot figures", action="store_true"
)
parser.add_argument(
    "-sf", "--storeFigure", help="store figures", action="store_true"
)

args = parser.parse_args()
print(args)

# == CONFIGURATION ==
env_name = "prob_zermelo_show_curvy-v0" # TODO change this for original environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
# print("MAX UPDATES: {}".format(maxUpdates))
# maxUpdates = 40
updateTimes = args.updateTimes
# print("UPDATE TIMES: {}".format(updateTimes))
# updateTimes = 10
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 250
# maxSteps = 5
storeFigure = args.storeFigure
plotFigure = args.plotFigure

if args.mode == 'lagrange':
  fn = args.name + '-' + args.doneType + '-' + args.costType
else:
  fn = args.name + '-' + args.doneType
if args.showTime:
  fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'naive', 'DDQN', args.mode, fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)

if args.mode == 'lagrange':
  envMode = 'normal'
  agentMode = 'normal'
  GAMMA_END = args.gamma
  EPS_PERIOD = updatePeriod
  EPS_RESET_PERIOD = maxUpdates
elif args.mode == 'RA':
  envMode = 'RA'
  agentMode = 'RA'
  if args.annealing:
    GAMMA_END = 0.999999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
  else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

if args.doneType == 'toEnd':
  sample_inside_obs = True
elif args.doneType == 'TF' or args.doneType == 'fail':
  sample_inside_obs = False

# == Environment ==
print("\n== Environment Information ==")
env = gym.make(
    env_name, device=device, mode=envMode, doneType=args.doneType,
    sample_inside_obs=sample_inside_obs, envType='basic'
)

stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print(
    "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
        stateDim, actionNum
    )
)
print(env.discrete_controls)

env.set_costParam(args.penalty, args.reward, args.costType, args.scaling)
env.set_seed(args.randomSeed)
print(
    "Cost type: {}, Margin scaling: {:.1f}, ".
    format(env.costType, env.scaling)
    + "Reward: {:.1f}, Penalty: {:.1f}".format(env.reward, env.penalty)
)

if plotFigure or storeFigure:
  nx, ny = 101, 101
  vmin = -1 * args.scaling
  vmax = 1 * args.scaling

  v = np.zeros((nx, ny))
  l_x = np.zeros((nx, ny))
  g_x = np.zeros((nx, ny))
  xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
  ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

  it = np.nditer(v, flags=['multi_index'])

  while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    l_x[idx] = env.target_margin(np.array([x, y]))
    g_x[idx] = env.safety_margin(np.array([x, y]))

    v[idx] = np.maximum(l_x[idx], g_x[idx])
    it.iternext()

  axStyle = env.get_axes()

  fig, axes = plt.subplots(1, 3, figsize=(12, 6))

  ax = axes[0]
  im = ax.imshow(
      l_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$\ell(x)$', fontsize=18)

  ax = axes[1]
  im = ax.imshow(
      g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$g(x)$', fontsize=18)

  ax = axes[2]
  im = ax.imshow(
      v.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
#   env.plot_reach_avoid_set(ax)
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$v(x)$', fontsize=18)

  for ax in axes:
    env.plot_target_failure_set(ax=ax)
    env.plot_formatting(ax=ax)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'env.png')
    fig.savefig(figurePath)
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

# == Agent CONFIG ==
print("\n== Agent Information ==")
CONFIG = dqnConfig(
    DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=100
)

# == AGENT ==
dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNSingle(
    CONFIG, actionNum, action_list, dimList=dimList, mode=agentMode,
    terminalType=args.terminalType
)
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

# Load model
updates = 380000
agent.restore(updates, r"/home/jambrown/RL_Learning/safety_rl/experiments/naive/DDQN/RA/-toEnd")

# Simulate trajectories on new environment
results = env.simulate_trajectories(
    agent.Q_network, T=maxSteps, num_rnd_traj=1000,
    toEnd=False
)[1]
success = np.sum(results == 1) / results.shape[0]
failure = np.sum(results == -1) / results.shape[0]
unfinish = np.sum(results == 0) / results.shape[0]

lr = agent.optimizer.state_dict()["param_groups"][0]["lr"]
print("  - eps={:.2f}, gamma={:.6f}, lr={:.1e}.".format(agent.EPSILON, agent.GAMMA, lr))
print("  - success/failure/unfinished ratio:", end=" ")
with np.printoptions(formatter={"float": "{: .3f}".format}):
    print(np.array([success, failure, unfinish]))

env.visualize(agent.Q_network, vmin=vmin, vmax=vmax, cmap="seismic", addBias=False)
figurePath = os.path.join(figureFolder, "Curvy_StateExtension_{:d}.png".format(updates))
plt.savefig(figurePath)