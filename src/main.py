import gymnasium as gym
import train.train_room as train_room

import environment.class_GridWorldEnv as class_GridWorldEnv
import agent.class_GridWorldAgent as agent_class

from agent.policy.qtable.class_PolicyQtable import PolicyQtable
from agent.policy.random.class_PolicyRandom import PolicyRandom
from play import playground
from agent.policy import policy_factory
from config import AppCfg


gym_env_id = 'gymnasium_env/GridWorld-v0'

def train_and_test():
    env = gym.make(gym_env_id, size=7)

    policy_explotate = None
    if AppCfg.POLICY_NAME_EXPLOTATE == 'dqn':
        policy_explotate = policy_factory.create_policy_dqn(env.action_space.n)
    elif AppCfg.POLICY_NAME_EXPLOTATE == 'qtable':
        policy_explotate = policy_factory.create_policy_qtable(env.action_space.n)

    policy_explorate = policy_factory.create_policy_random(env.action_space.n)

    agent = agent_class.GridWorldAgent(policy_explotate, policy_explorate)

    train_room.train(env, agent)

    # manual test
    env = gym.make(gym_env_id, size=7, render_mode="human")
    agent.eval_mode()

    playground.play(env, agent)
    env.close()


def play_only(play_count: int, policy_file: str):
    env = gym.make(gym_env_id, size=7, render_mode="human")

    policy_explotate = None
    if AppCfg.POLICY_NAME_EXPLOTATE == 'dqn':
        policy_explotate = policy_factory.create_policy_dqn(env.action_space.n)
    elif AppCfg.POLICY_NAME_EXPLOTATE == 'qtable':
        policy_explotate = policy_factory.create_policy_qtable(env.action_space.n)

    policy_explorate = policy_factory.create_policy_random(env.action_space.n)
    policy_explotate.import_from_file(policy_file)

    agent = agent_class.GridWorldAgent(policy_explotate, policy_explorate)
    agent.eval_mode()

    for i in range(0, play_count):
        playground.play(env, agent)

    env.close()


def get_play_path() -> str:
    if AppCfg.POLICY_NAME_EXPLOTATE == 'dqn':
        return '__target/grid_world_policy_dqn_100.pth'
    elif AppCfg.POLICY_NAME_EXPLOTATE == 'qtable':
        return '__target/grid_world_policy_qtable_100.json'
    
    raise Exception('Error to get play path. Please, check AppCfg.POLICY_NAME_EXPLOTATE')


def main() -> int:
    class_GridWorldEnv.register()

    if AppCfg.EXECUTION_MODE == 'train':
        train_and_test()
    elif AppCfg.EXECUTION_MODE == 'play':
        play_only(AppCfg.PLAY_COUNT, get_play_path())

    return 0


if __name__ == '__main__':
    main()
