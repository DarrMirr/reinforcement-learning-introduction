import numpy as np
import random

from agent.policy.interface_AbstractPolicy import AbstractPolicy
from util.logger import log

class PolicyRandom(AbstractPolicy):


    def __init__(self, env_action_num: int):
        super().__init__()
        self.env_action_num: int = env_action_num


    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        num_actions = self.env_action_num
        next_action = random.randrange(num_actions)
        log.debug(f'exploration: choose random action = {next_action}')

        return next_action
