import gymnasium as gym
from gymnasium.wrappers import FilterObservation, FlattenObservation
from envs.lowcostrobot import LowCostRobotWrapper

LOWCOSTROBOT_TASKS = {
    'MyPushCube-v0': 'MyPushCube-v0',
    'MyReachCube-v0': 'MyReachCube-v0',
}

def make_env(cfg):
    if not cfg.task in LOWCOSTROBOT_TASKS:
        raise ValueError('Unknown task:', cfg.task)

    env = gym.make(
        f"gym_lowcostrobot:{LOWCOSTROBOT_TASKS[cfg.task]}",
        max_episode_steps=cfg.env["max_episode_steps"],
        observation_cameras=cfg.env["observation_cameras"],
        render_mode='rgb_array'
        )
    
    env = FilterObservation(env, cfg.env["filter_keys"] + [f'image_{cam}' for cam in env.cameras])
    env = FlattenObservation(env)
    env = LowCostRobotWrapper(env)
    
    env.max_episode_steps = env.spec.max_episode_steps
    # cfg.discount_max = 0.97 #0.99=tdmpc; 0.9=lerobot
    cfg.rho = 0.7 #0.5=lerobot # TODO: increase rho for episodic tasks since termination always happens at the end of a sequence
    return env