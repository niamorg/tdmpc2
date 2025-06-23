import gymnasium as gym
from gymnasium.wrappers import FilterObservation, FlattenObservation

LOWCOSTROBOT_TASKS = {
    'PushCube-v0': 'PushCube-v0',
}

class LowCostRobotWrapper(gym.Wrapper):
    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # info['terminated'], info['truncated'] = terminated, truncated
        return obs, reward, terminated or truncated, info
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)


def make_env(cfg):
    if not cfg.task in LOWCOSTROBOT_TASKS:
        raise ValueError('Unknown task:', cfg.task)

    envcfg = cfg.lowcostrobot

    env = gym.make(
        f"gym_lowcostrobot:{LOWCOSTROBOT_TASKS[cfg.task]}",
        **envcfg["kwargs"],
        )
    
    filter_keys = {None: [],
                   'joint': ['arm_qpos', 'arm_qvel'],
                   'ee': ['ee_xpos', 'ee_xvel'],
                   'all': ['arm_qpos', 'arm_qvel', 'ee_xpos', 'ee_xvel']}[envcfg["robot_observation_mode"]]

    filter_keys += ['cube_pos'] if envcfg["kwargs"]["observation_mode"] == 'state' else ['image_front', 'image_top']
    filter_keys += ['target_pos'] if (cfg.task == "PushCube-v0" and envcfg["kwargs"]["observation_mode"] == 'state') else []
    filter_keys += ['cube_vel'] if envcfg["cube_vel"] else []

    env = FilterObservation(env, filter_keys)
    env = FlattenObservation(env)
    env = LowCostRobotWrapper(env)
    
    cfg.discount_max = 0.97 #0.99=tdmpc; 0.9=lerobot
    cfg.rho = 0.7 #0.5=lerobot # TODO: increase rho for episodic tasks since termination always happens at the end of a sequence
    return env