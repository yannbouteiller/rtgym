from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone
import gym.spaces as spaces
import gym
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

    def reset(self):
        if not self.initialized:
            self.rc_drone = DummyRCDrone()
            self.initialized = True
        pos_x, pos_y = self.rc_drone.get_observation()
        self.target[0] = np.random.uniform(-0.5, 0.5)
        self.target[1] = np.random.uniform(-0.5, 0.5)
        return [pos_x, pos_y, self.target[0], self.target[1]]

    def get_obs_rew_done_info(self):
        pos_x, pos_y = self.rc_drone.get_observation()
        tar_x = self.target[0]
        tar_y = self.target[1]
        obs = [pos_x, pos_y, tar_x, tar_y]
        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
        done = rew > -0.01
        info = {}
        return obs, rew, done, info

    def wait(self):
        self.send_control(self.get_default_action())

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

env = gym.make("rtgym:real-time-gym-v0", config=my_config)


def model(obs):
    return np.array([obs[2] - obs[0], obs[3] - obs[1]], dtype=np.float32) * 20.0


done = False
obs = env.reset()
while not done:
    env.render()
    act = model(obs)
    obs, rew, done, info = env.step(act)
    print(f"rew:{rew}")

print("Environment benchmarks:")
pprint.pprint(env.benchmarks())

cv2.waitKey(0)
