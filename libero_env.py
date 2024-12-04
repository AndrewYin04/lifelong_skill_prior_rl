import os
import torch
import numpy as np
import cv2
import gym

from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv

task_suites = ["libero_object", "libero_spatial", "libero_goal", "libero_10", "libero_90", "libero_100"]

class CFG:
    def __init__(self, task_suite, task_id, obs):
        self.task_suite = task_suite
        self.task_id = task_id
        self.obs = obs

class LIBEROEnvWrapper():
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_episode_steps = 200
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = self.benchmark_dict[cfg.task_suite]()
        self.task_id = cfg.task_id
        
        self.env = self.get_env(self.task_id)
        
        self.action_space = gym.spaces.Box(
			low=np.full(7, -1),
			high=np.full(7, 1),
			dtype=np.float32)

        self.t = 0
        # from gym.spaces import Dict
        # breakpoint()
        # self.observation_space = Dict(self.env.env.observation_spec())
        self.observation_space = self.env.env.observation_spec()
        self.action_space = self.env.env.action_spec

    def get_env(self, task_idx):
        self.task = self.task_suite.get_task(task_idx)
        self.task_description = self.task.language
        self.task_bddl_file = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        print(f"[info] retrieving task {self.task_id} from suite {self.cfg.task_suite}, the " + \
            f"language instruction is {self.task_description}, and the bddl file is {self.task_bddl_file}")
        
        self.hdf5_filename = f"libero/libero/bddl_files/libero_spatial/{self.task.name}_demo.hdf5"
        print(f"HDF5 Filename: {self.hdf5_filename}")

        self.env_args = {
            "has_renderer" : True,
            "ignore_done" : True,
            "horizon" : self.max_episode_steps,
            "bddl_file_name": self.task_bddl_file,
            "camera_heights": 64,
            "camera_widths": 64
        }

        env = OffScreenRenderEnv(**self.env_args)
        env.seed(0)
        env.reset()
        self.init_states = self.task_suite.get_task_init_states(self.task_id)
        self.init_state_id = 0
        env.set_init_state(self.init_states[self.init_state_id])

        return env

    def reset(self, **kwargs):
        self.t = 0
        self.env.reset()
        dummy_action = [0.] * 7
        for _ in range(10):
            self.obs, self.reward, self.done, self.info = self.env.step(dummy_action)
            if self.done:
                break

        return self.obs

    def step(self, action):
        assert len(action) == 7, "Action length must be 7"
        self.t += 1
        self.obs, self.reward, self.done, self.info = self.env.step(np.array(action))
        return self.obs, self.reward - 1.0, self.done or self.t == self.max_episode_steps, self.info

    def render(self, *args, **kwargs):
        height = kwargs.get('height', 64)
        width = kwargs.get('height', 64)
        return np.array(self.obs["agentview_image"])
def make_env(cfg):

    assert cfg.task_suite in task_suites, (f"{cfg.task_suite} not a valid task. Options are libero_object, libero_spatial, libero_goal, libero_10, libero_100")
    assert cfg.obs == 'rgb', 'This task only supports rgb observations.'
    return LIBEROEnvWrapper(cfg=cfg)