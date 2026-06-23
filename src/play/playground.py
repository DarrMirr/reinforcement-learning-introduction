import gymnasium as gym

from util.logger import log
from agent.class_GridWorldAgent import GridWorldAgent

def play(env: gym.Env, agent: GridWorldAgent):
    log.info('Start play session')

    terminated = False
    step_count = 0
    obs, info = env.reset()
    while not terminated and step_count < 30:
        step_count += 1
        log.info(f'---[step={step_count}]-------------------------')
        
        action = agent.get_action(obs)
        log.info(f'choosen action={action.num}, reason={action.meta_reason}')

        obs, reward, terminated, truncated, info = env.step(action.num)
        log.info(f'after action observation={obs}')
        log.info(f'after action reward={reward}')

    log.info(f'Finish play session : steps={step_count}, is_successfully_terminated={terminated}')