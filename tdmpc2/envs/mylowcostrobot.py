import gymnasium as gym
from gymnasium.wrappers import FilterObservation, FlattenObservation

LOWCOSTROBOT_TASKS = ['MyPushCube-v0', 'MyReachCube-v0']

class TerminatedTruncatedToDone(gym.Wrapper):
    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['terminated'], info['truncated'] = terminated, truncated
        return obs, reward, terminated or truncated, info


def make_env(cfg):
    if not cfg.task in LOWCOSTROBOT_TASKS:
        raise ValueError('Unknown task:', cfg.task)

    env = gym.make(f"gym_lowcostrobot:{cfg.task}", **cfg.env["kwargs"], render_mode='rgb_array', disable_env_checker=True)
    
    env = FilterObservation(env, cfg.env["filter_keys"] + [f'image_{cam}' for cam in env.unwrapped.cameras])
    env = FlattenObservation(env)
    env = TerminatedTruncatedToDone(env)
    
    env.max_episode_steps = env.spec.max_episode_steps
    # cfg.discount_max = 0.97 #0.99=tdmpc; 0.9=lerobot
    cfg.rho = 0.7 #0.5=lerobot # TODO: increase rho for episodic tasks since termination always happens at the end of a sequence
    return env