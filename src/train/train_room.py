import gymnasium as gym
import os
import util.device as device
import train.train_visualizer as train_visualizer

from tqdm import tqdm
from util.logger import log
from agent.class_GridWorldAgent import GridWorldAgent
from agent.class_GridWorldAgent import AgentAction
from train.class_train_metrics import TrainMetrics

from config import TrainCfg



def train(env: gym.Env, agent: GridWorldAgent):
    # TRAINING HYPERPARAMETERS
    episode_num = TrainCfg.EPISODE_NUM
    episode_step_count = TrainCfg.EPISODE_STEP_COUNT

    log.info(f'Start train session : total_episodes={episode_num}, device={device.accelerator}')
    train_metrics = TrainMetrics(rate=TrainCfg.METRICS_RATE)
    seed = 1
    for episode in tqdm(range(episode_num)):
        train_metrics.start_new_episode(episode)
        if episode % TrainCfg.ENV_SEED_RATE == 0:
            seed += 1

        obs, info = env.reset(seed=seed)

        for step in range(episode_step_count):
            train_metrics.episode_metrics().inc_episode_step_num()

            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action.num)

            agent.memorize_exp(action.num, reward, obs)
            loss_output = agent.learn()

            train_metrics.episode_metrics().append_loss_fn_output(loss_output)

            is_positive_exp = agent.is_positive_last_exp()
            if action.meta_reason == AgentAction.META_REASON_EXPLORATION:
                train_metrics.episode_metrics().inc_exploration_action_num(
                    is_positive_exp)
            elif action.meta_reason == AgentAction.META_REASON_EXPLOTATION:
                train_metrics.episode_metrics().inc_explotation_action_num(
                    is_positive_exp)

            if terminated:
                log.debug(
                    f'Training game session successfully terminated : episode={episode}, turn={step}'
                )
                train_metrics.episode_metrics(
                ).mark_episode_as_successfully_terminated()
                break

            if truncated:
                log.debug(
                    f'Training game session truncated : episode={episode}, turn={step}'
                )
                break

        train_metrics.end_current_episode()

    train_visualizer.visualize(train_metrics)
    __export_policy(agent, episode_num)
    log.info(f'End train session : total_episodes={episode_num}')


def __export_policy(agent: GridWorldAgent, episode_num: int):
    target_dir = '__target'
    os.makedirs(target_dir, exist_ok=True)
    agent.policy_explotate.export_to_file(target_dir=target_dir, file_prefix='grid_world', file_suffix=str(episode_num))
