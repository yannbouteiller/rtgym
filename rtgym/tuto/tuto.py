from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone
import gymnasium.spaces as spaces
import gymnasium
import numpy as np
import cv2

import time
import pprint


def test_rc_drone():
    rc_drone = DummyRCDrone()
    for i in range(10):
        if i < 5:  # first 5 iterations
            vel_x = 0.1
            vel_y = 0.5
        else:  # last 5 iterations
            vel_x = 0.0
            vel_y = 0.0
        rc_drone.send_control(vel_x, vel_y)
        pos_x, pos_y = rc_drone.get_observation()
        print(f"iteration {i}, sent vel: vel_x:{vel_x}, vel_y:{vel_y} - received pos: x:{pos_x:.3f}, y:{pos_y:.3f}")
        time.sleep(0.05)


class MyRealTimeInterface(RealTimeGymInterface):

    def __init__(self):
        self.rc_drone = None
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.initialized = False

    def get_observation_space(self):
        pos_x_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        pos_y_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        tar_x_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        tar_y_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        return spaces.Tuple((pos_x_space, pos_y_space, tar_x_space, tar_y_space))

    def get_action_space(self):
        return spaces.Box(low=-2.0, high=2.0, shape=(2,))

    def get_default_action(self):
        return np.array([0.0, 0.0], dtype='float32')

    def send_control(self, control):
        vel_x = control[0]
        vel_y = control[1]
        self.rc_drone.send_control(vel_x, vel_y)

    def reset(self, seed=None, options=None):
        if not self.initialized:
            self.rc_drone = DummyRCDrone()
            self.initialized = True
        pos_x, pos_y = self.rc_drone.get_observation()
        self.target[0] = np.random.uniform(-0.5, 0.5)
        self.target[1] = np.random.uniform(-0.5, 0.5)
        return [np.array([pos_x], dtype='float32'),
                np.array([pos_y], dtype='float32'),
                np.array([self.target[0]], dtype='float32'),
                np.array([self.target[1]], dtype='float32')], {}

    def get_obs_rew_terminated_info(self):
        pos_x, pos_y = self.rc_drone.get_observation()
        tar_x = self.target[0]
        tar_y = self.target[1]
        obs = [np.array([pos_x], dtype='float32'),
               np.array([pos_y], dtype='float32'),
               np.array([tar_x], dtype='float32'),
               np.array([tar_y], dtype='float32')]
        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
        terminated = rew > -0.01
        info = {}
        return obs, rew, terminated, info

    def wait(self):
        pass

    def render(self):
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        pos_x, pos_y = self.rc_drone.get_observation()
        image = cv2.circle(img=image,
                           center=(int(pos_x * 200) + 200, int(pos_y * 200) + 200),
                           radius=10,
                           color=(255, 0, 0),
                           thickness=1)
        image = cv2.circle(img=image,
                           center=(int(self.target[0] * 200) + 200, int(self.target[1] * 200) + 200),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)
        cv2.imshow("PipeLine", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


my_config = DEFAULT_CONFIG_DICT
my_config["interface"] = MyRealTimeInterface
my_config["time_step_duration"] = 0.05
my_config["start_obs_capture"] = 0.05
my_config["time_step_timeout_factor"] = 1.0
my_config["ep_max_length"] = 100
my_config["act_buf_len"] = 4
my_config["reset_act_buf"] = False
my_config["benchmark"] = True
my_config["benchmark_polyak"] = 0.2

env = gymnasium.make("real-time-gym-v1", config=my_config)

obs_space = env.observation_space
act_space = env.action_space

print("==============================")
print(f"Observation space:\n{obs_space}")
print(f"Action space:\n{act_space}")
print("==============================")


def model(obs):
    return np.clip(np.concatenate((obs[2] - obs[0], obs[3] - obs[1])) * 20.0, -2.0, 2.0)


terminated, truncated = False, False
obs, info = env.reset()
while not (terminated or truncated):
    env.render()
    act = model(obs)
    obs, rew, terminated, truncated, info = env.step(act)
    print(f"rew:{rew}")

if terminated:
    print(f"Task complete.")
elif truncated:
    print(f"Episode truncated due to time-steps limit.")

print("Environment benchmarks:")
pprint.pprint(env.benchmarks())

cv2.waitKey(0)
