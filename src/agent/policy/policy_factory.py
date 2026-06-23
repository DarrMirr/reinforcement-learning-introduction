import agent.policy.dqn.class_DQN as agent_policy

from agent.policy.dqn.class_PolicyDQN import PolicyDQN
from agent.policy.qtable.class_PolicyQtable import PolicyQtable
from agent.policy.random.class_PolicyRandom import PolicyRandom
from config import PolicyCfgDQN



def create_policy_dqn(env_action_num: int) -> PolicyDQN:
    input_count = PolicyCfgDQN.INPUT_COUNT
    hidden_layer_size = PolicyCfgDQN.HIDDEN_LAYER_SIZE

    hidden_layer_size_tuple = (hidden_layer_size, hidden_layer_size)
    policy_net = agent_policy.DQN(input_count, env_action_num, hidden_layer_size_tuple)

    return PolicyDQN(policy_net)


def create_policy_random(env_action_num: int) -> PolicyRandom:
    return PolicyRandom(env_action_num)
    

def create_policy_qtable(env_action_num: int) -> PolicyQtable:
    return PolicyQtable(env_action_num)