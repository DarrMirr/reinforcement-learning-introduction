import util.device as device
import torch
import torch.nn as nn
import numpy as np
import os

from agent.class_AgentMemory import AgentMemory
from util.logger import log
from agent.policy.interface_AbstractPolicyLearnable import AbstractPolicyLearnable


class PolicyDQN(AbstractPolicyLearnable):
    """
    Policy implementation of simplified DQN algorithm. "Simplified" means lack of using target neural network.

    Parameters:
        policy_net (nn.Module): Policy neural net.

    Attributes:
        optimizer (torch.optim.Adam): Optimizer.
        loss_func (nn.MSELoss): Loss function.

    Author:
        Darr Mirr (Vladimir S. Polukeev)
    """

    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=0.01)
        self.loss_func = nn.MSELoss()

    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        # we do not want to gather gradients as we are only generating experience, not training the network
        with torch.no_grad():
            policy_input = self.__to_policy_input_obs(obs)
            policy_output = self.policy_net(policy_input)
            next_action = policy_output.argmax()
            log.debug(
                f'explotation: choose the best fit action = {next_action}')

            return next_action.item()

    def learn(self, train_samples: list[tuple]) -> float:
        input_tensor_stack_before, action_tensor_stack = self.__get_input_and_reward_tensors(
            train_samples)

        self.optimizer.zero_grad()

        actions_predict = self.policy_net(input_tensor_stack_before)
        actions_target = self.__to_target_actions(actions_predict,
                                                  action_tensor_stack)

        loss_output = self.loss_func(actions_predict, actions_target)
        loss_output.backward()
        self.optimizer.step()

        return loss_output.item()

    def export_to_file(self,
                       target_dir: str,
                       file_prefix: str = '',
                       file_suffix: str = ''):
        file_name = '_'.join([file_prefix, 'policy_dqn',
                              file_suffix]).replace('_$', '')
        file_name = file_name + '.pth'
        file_path = os.path.join(target_dir, file_name)
        torch.save(self.policy_net.state_dict(), file_path)

    def import_from_file(self, file_path: str):
        import_state_dict = torch.load(file_path, weights_only=True)
        self.policy_net.load_state_dict(import_state_dict)

    def __to_policy_input_obs(self, obs: dict[str,
                                              np.ndarray]) -> torch.Tensor:
        target = obs['target']
        target_direction_row = target[0]
        target_direction_col = target[1]

        return self.__to_policy_input(target_direction_row,
                                      target_direction_col)

    def __to_policy_input(self, target_direction_col: int,
                          target_direction_row: int) -> torch.Tensor:
        np_array = np.array((target_direction_col, target_direction_row))

        return torch.FloatTensor(np_array)

    def __get_input_and_reward_tensors(
            self, memory_sample_tuples: list[tuple]) -> tuple[torch.Tensor]:
        input_tensor_tuple_before = []
        action_tensor_tuple = []

        for sample_tuple in memory_sample_tuples:
            # s (state before step)
            target_direction_row = getattr(
                sample_tuple, AgentMemory.SAMPLE_TARGET_ROW_D_BEFORE)
            target_direction_col = getattr(
                sample_tuple, AgentMemory.SAMPLE_TARGET_COL_D_BEFORE)
            input_tensor = self.__to_policy_input(target_direction_row,
                                                  target_direction_col)
            input_tensor_tuple_before.append(input_tensor)

            action = getattr(sample_tuple, AgentMemory.SAMPLE_ACTION)
            action_tensor = torch.IntTensor([action]).to(device.accelerator)
            action_tensor_tuple.append(action_tensor)

        input_tensor_stack_before = torch.stack(input_tensor_tuple_before)
        action_tensor_stack = torch.stack(action_tensor_tuple)

        return input_tensor_stack_before, action_tensor_stack

    def __to_target_actions(self, q_values: torch.Tensor,
                            action_tensor: torch.Tensor) -> torch.Tensor:
        mask_negative = torch.zeros_like(q_values, dtype=torch.bool)
        for i, idx in enumerate(action_tensor):
            mask_negative[i, idx] = True

        mask_positive = torch.logical_not(mask_negative)

        q_values_target = torch.zeros_like(q_values)
        q_values_target = q_values_target.where(mask_negative,
                                                torch.tensor(-1.0))
        q_values_target = q_values_target.where(mask_positive,
                                                torch.tensor(1.0))

        return q_values_target
