import numpy as np
import json
import os
import random

from collections import deque
from agent.policy.interface_AbstractPolicyLearnable import AbstractPolicyLearnable
from agent.class_AgentMemory import AgentMemory


class PolicyQtable(AbstractPolicyLearnable):
    """
    Policy implementation of Q-table algorithm.

    Parameters:
        env_action_num (int): Amount of Environment action space.

    Attributes:
        qtable_dict (dict): Q-table storage.

    Author:
        Darr Mirr (Vladimir S. Polukeev)
    """
    
    def __init__(self, env_action_num: int):
        super().__init__()
        self.env_action_num: int = env_action_num
        self.qtable_dict: dict[str, deque[int]] = {}


    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        target = obs['target']
        key = self.__create_key(target[0], target[1])

        action_deque = self.qtable_dict[key]
        if len(action_deque) > 0:
            action = action_deque.popleft()
            action_deque.append(action)

            return action
        else:
            num_actions = self.env_action_num
            next_action = random.randrange(num_actions)
            
            return next_action
        

    def __create_key(self, target_direction_row: int, target_direction_col: int) -> str:
        key = str(target_direction_row) + str(target_direction_col)
        if key not in self.qtable_dict:
            self.qtable_dict[key] = deque()

        return key

    
    def learn(self, train_samples: list[tuple]) -> float:
        for sample_tuple in train_samples:
            target_direction_row = getattr(sample_tuple, AgentMemory.SAMPLE_TARGET_ROW_D_BEFORE)
            target_direction_col = getattr(sample_tuple, AgentMemory.SAMPLE_TARGET_COL_D_BEFORE)
            action = getattr(sample_tuple, AgentMemory.SAMPLE_ACTION)

            key = self.__create_key(target_direction_row, target_direction_col)
            action_deque = self.qtable_dict[key]
            if action not in action_deque:
                action_deque.append(action)

    
    def export_to_file(self, target_dir: str, file_prefix: str = '', file_suffix: str = ''):
        file_name = '_'.join([file_prefix, 'policy_qtable' , file_suffix]).replace('_$', '')
        file_name = file_name + '.json' 
        file_path = os.path.join(target_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.qtable_dict, f, cls=DequeEncoder, ensure_ascii=False, indent=4)
    

    def import_from_file(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.qtable_dict = json.load(f, cls=DequeDecoder)


class DequeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, deque):
            return list(obj)
        return super().default(obj)
    

class DequeDecoder(json.JSONDecoder):
     def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

     def object_hook(self, obj):
        if isinstance(obj, list):
            return self._convert_to_deque(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_deque(value) for key, value in obj.items()}
        else:
            return obj

     def _convert_to_deque(self, item):
        if isinstance(item, list):
            return deque(item)
        else:
            return item
