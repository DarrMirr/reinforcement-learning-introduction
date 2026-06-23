import numpy as np
import random

from collections import deque, namedtuple


class AgentMemory():
    """
    Parameters:
        capacity (int): The maximum number of experiences that can be stored in memory.

    Attributes:
        memory_new_exp (deque): A deque containing the experiences.
        memory_positive_dict (dict): A dictionary of positive experience.

    Author:
        Darr Mirr (Vladimir S. Polukeev)
    """

    # Define a named tuple to store the experiences
    SAMPLE_ACTION = 'action'
    SAMPLE_REWARD = 'reward'
    SAMPLE_AGENT_ROW_N_BEFORE = 's_agent_row_num'  # before step
    SAMPLE_AGENT_COL_N_BEFORE = 's_agent_col_num'  # before step
    SAMPLE_TARGET_ROW_D_BEFORE = 's_target_row_dir'  # before step
    SAMPLE_TARGET_COL_D_BEFORE = 's_target_col_dir'  # before step
    SAMPLE_AGENT_ROW_N_AFTER = 's_prime_agent_row_num'  # after step
    SAMPLE_AGENT_COL_N_AFTER = 's_prime_agent_col_num'  # after step
    SAMPLE_TARGET_ROW_D_AFTER = 's_prime_target_row_dir'  # after step
    SAMPLE_TARGET_COL_D_AFTER = 's_prime_target_col_dir'  # after step
    SAMPLE_IS_POSITIVE_EXP = 'is_positive_exp'

    Sample = namedtuple(
        "Sample", (SAMPLE_ACTION, SAMPLE_REWARD, SAMPLE_IS_POSITIVE_EXP,
                   SAMPLE_AGENT_ROW_N_BEFORE, SAMPLE_AGENT_COL_N_BEFORE,
                   SAMPLE_TARGET_ROW_D_BEFORE, SAMPLE_TARGET_COL_D_BEFORE,
                   SAMPLE_AGENT_ROW_N_AFTER, SAMPLE_AGENT_COL_N_AFTER,
                   SAMPLE_TARGET_ROW_D_AFTER, SAMPLE_TARGET_COL_D_AFTER))
    __EMPTY_SAMPLE = Sample(-1, -1, False, -1, -1, -1, -1, -1, -1, -1, -1)

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory_new_exp = deque([], maxlen=capacity)
        self.memory_positive_dict = dict()  # key=action, value=deque[Sample]

    def save(self, action: int, reward: float, obs: dict[str, np.ndarray]):
        sample = self.__convert_to_sample(action, reward, obs)

        self.memory_new_exp.append(sample)

        is_positive_exp = getattr(sample, self.SAMPLE_IS_POSITIVE_EXP)
        if (is_positive_exp):
            if action not in self.memory_positive_dict:
                self.memory_positive_dict[action] = deque([],
                                                          maxlen=self.capacity)

            self.memory_positive_dict[action].append(sample)

    def __convert_to_sample(self, action: int, reward: float,
                            obs: dict[str, np.ndarray]) -> Sample:
        agent = obs['agent']
        target = obs['target']

        agent_row_n_after = agent[0]
        agent_col_n_after = agent[1]
        target_row_d_after = target[0]
        target_col_d_after = target[1]

        prev_sample = self.get_last_sample()
        agent_row_n_before = getattr(prev_sample,
                                     self.SAMPLE_AGENT_ROW_N_AFTER)
        agent_col_n_before = getattr(prev_sample,
                                     self.SAMPLE_AGENT_COL_N_AFTER)
        target_row_d_before = getattr(prev_sample,
                                      self.SAMPLE_TARGET_ROW_D_AFTER)
        target_col_d_before = getattr(prev_sample,
                                      self.SAMPLE_TARGET_COL_D_AFTER)
        positive_exp = agent_row_n_before != -1 and reward > getattr(
            prev_sample, self.SAMPLE_REWARD)

        return AgentMemory.Sample(action, reward, positive_exp,
                                  agent_row_n_before, agent_col_n_before,
                                  target_row_d_before, target_col_d_before,
                                  agent_row_n_after, agent_col_n_after,
                                  target_row_d_after, target_col_d_after)

    def load_sample_random_positive_balanced(
            self, max_batch_size: int) -> list[tuple]:
        sizeof_smallest_action_samples = self.sizeof_smallest_action_samples()
        batch_size = max_batch_size if max_batch_size <= sizeof_smallest_action_samples else sizeof_smallest_action_samples

        balanced_sample_positive_list = []
        for action_samples_deque in self.memory_positive_dict.values():
            action_sample_list = random.sample(action_samples_deque,
                                               batch_size)
            balanced_sample_positive_list.extend(action_sample_list)

        return balanced_sample_positive_list

    def sizeof_smallest_action_samples(self) -> int:
        smallest_size = -1
        for action_samples_deque in self.memory_positive_dict.values():
            len_deque = len(action_samples_deque)
            if smallest_size == -1 or len_deque < smallest_size:
                smallest_size = len_deque

        return 0 if smallest_size == -1 else smallest_size

    def get_last_sample(self) -> tuple:
        if len(self.memory_new_exp) == 0:
            return AgentMemory.__EMPTY_SAMPLE
        else:
            return self.memory_new_exp[-1]

    def len_positive(self) -> int:
        return len(self.memory_positive_dict)

    def len_new_exp_memory(self) -> int:
        return len(self.memory_new_exp)

    def clear_new_exp_memory(self):
        self.memory_new_exp.clear()
