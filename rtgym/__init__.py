from rtgym.envs.real_time_env import RealTimeGymInterface, DEFAULT_CONFIG_DICT
from rtgym.tuto.dummy_drone import DummyRCDrone
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='real-time-gym-v0',
    entry_point='rtgym.envs:RealTimeEnv',
)
