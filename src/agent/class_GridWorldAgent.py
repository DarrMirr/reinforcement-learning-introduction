import numpy as np
import random
import math

from util.logger import log
from agent.class_AgentMemory import AgentMemory
from agent.policy.interface_AbstractPolicyLearnable import AbstractPolicyLearnable
from agent.policy.interface_AbstractPolicy import AbstractPolicy
from config import AgentCfg


class GridWorldAgent():
    """
    Parameters:
        policy_explotate (AbstractPolicyLearnable): Policy to grasp maximum reward from Environment action.
        policy_explorate (AbstractPolicy): Policy to discover Environment and collect experience.

    Attributes:
        memory (AgentMemory): Experience storage.
        learn_batch_size (int): Amount of experience samples requested from Agent memory.
        is_train_mode (bool): Toggle flag to turn on/off Agent learn capability.

    Author:
        Darr Mirr (Vladimir S. Polukeev)
    """

    def __init__(self, policy_explotate: AbstractPolicyLearnable,
                 policy_explorate: AbstractPolicy):
        self.policy_explotate: AbstractPolicyLearnable = policy_explotate
        self.policy_explorate: AbstractPolicy = policy_explorate

        self.memory = AgentMemory(AgentCfg.MEMORY_SIZE)
        self.learn_batch_size: int = AgentCfg.LEARN_BATCH_SIZE
        self.is_train_mode: bool = True

        # Exploration parameters for epsilon greedy strategy
        self.start_epsilon = AgentCfg.START_EPSILON
        self.stop_epsilon = AgentCfg.STOP_EPSILON
        self.decay_rate = AgentCfg.DECAY_RATE
        self.done_actions = 0

        log.info(f'Agent created : learn_batch_size={self.learn_batch_size}')

    """
      This method is step of agent action pipeline against Environment.
      Calling this method emulates agent's choice for the next action against Environment.

      Method consist of two parts:
      - Exploration (try new action)
      - Explotation (choose the best known action)
    """

    def get_action(self, obs: dict[str, np.ndarray]) -> "AgentAction":
        sample = 0
        eps_threshold = 1
        if self.is_train_mode:
            # generate a random number
            sample = random.random()

            # calculate the epsilon threshold, based on the epsilon-start value, the epsilon-stop value,
            # the number of training steps taken and the epsilon decay rate
            # here we are using an exponential decay rate for the epsilon value
            sizeof_sample_positive = self.memory.sizeof_smallest_action_samples(
            )
            done_action_factor = sizeof_sample_positive if sizeof_sample_positive > 0 else self.done_actions
            eps_threshold = self.stop_epsilon + (
                self.start_epsilon - self.stop_epsilon) * math.exp(
                    -1.0 * done_action_factor / self.decay_rate)

        if sample < eps_threshold:
            log.debug(f'sample={sample}, threshold={eps_threshold}')

            # explotation part:
            #     act greedily towards the Q-values of our policy network, given the state
            action_num = self.policy_explotate.get_action(obs)
            return AgentAction(action_num, AgentAction.META_REASON_EXPLOTATION)
        else:
            # exploration part:
            #     select a random action with equal probability
            action_num = self.policy_explorate.get_action(obs)
            return AgentAction(action_num, AgentAction.META_REASON_EXPLORATION)

    def memorize_exp(self, action: int, reward: float, obs: dict[str,
                                                                 np.ndarray]):
        self.memory.save(action, reward, obs)
        self.done_actions = self.done_actions + 1

    def is_positive_last_exp(self) -> bool:
        last_sample = self.memory.get_last_sample()
        return getattr(last_sample, AgentMemory.SAMPLE_IS_POSITIVE_EXP)

    def learn(self) -> float:
        if self.is_train_mode and self.memory.len_new_exp_memory(
        ) > self.learn_batch_size:
            memory_sample_tuples = self.memory.load_sample_random_positive_balanced(
                self.learn_batch_size)

            loss_output = self.policy_explotate.learn(memory_sample_tuples)
            self.memory.clear_new_exp_memory()

            return loss_output

    def eval_mode(self):
        self.is_train_mode = False

    def train_mode(self):
        self.is_train_mode = True


class AgentAction():
    META_REASON_EXPLORATION = 'exploration'
    META_REASON_EXPLOTATION = 'explotation'

    def __init__(self, num, meta_reason):
        self.num = num
        self.meta_reason = meta_reason
