import gym
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
from collections import deque
from gym import spaces
from .atari_wrappers import WarpFrame, ClipRewardEnv, FrameStack, ScaledFloatFrame, ChannelsFirstImageShape

class KaboomWrapper():
    @staticmethod
    def wrap(env):
        env = WarpFrame(env)
        # env = ChannelsFirstImageShape(env)
        env = FrameStack(env, 4)
        env = ClipRewardEnv(env)
        env = ScaledFloatFrame(env)
        return env



