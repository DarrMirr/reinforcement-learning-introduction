class AppCfg():
  POLICY_NAME_EXPLOTATE = 'dqn'         # dqn and qtable are available now
  EXECUTION_MODE = 'play'               # train and play are available now
  PLAY_COUNT = 5                        # play game session (For EXECUTION_MODE == 'play')


class TrainCfg():
  EPISODE_NUM = 100             # total episode count
  EPISODE_STEP_COUNT = 300      # total steps in each episode
  ENV_SEED_RATE = 5             # frequency of environment changes by episode

  METRICS_RATE = 1              # storing episode metrics rate (treat as store metric each METRICS_RATE)


class AgentCfg():
  LEARN_BATCH_SIZE = 200        # batch of samples retrieved from agent memory at learn stage
  START_EPSILON = 0.9           # exploration probability at start
  STOP_EPSILON = 0.05           # minimum exploration probability
  DECAY_RATE = 10000            # exponential decay rate for exploration prob
  MEMORY_SIZE = 5000            # amount of step sample to keep at agent memory


class PolicyCfgDQN():
  INPUT_COUNT = 2                         # input parameters count to policy network
  HIDDEN_LAYER_SIZE = 128                 # size of hidden layers at policy network
