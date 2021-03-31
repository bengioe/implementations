#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice, randint
#import ipdb
from time import sleep
from os import system
from pprint import PrettyPrinter
from six.moves import input
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image
from torchvision.transforms import ToTensor

# Height and width of the MazeBase environment
ENV_HEIGHT = 210
ENV_WIDTH = 160

class Number:
    def __init__(self, n):
        self.n = n

to_tensor_func = ToTensor()

class Env:

    """Docstring for Env. """

    def __init__(self, atari_env):
        self.action_space = atari_env.action_space #Number(atari_env.action_space.n)
        self.observation_space = atari_env.observation_space #Number(int(np.prod(self.observe().shape)))
        self.env = atari_env
        self.cur_obs = (self.env.reset())
        self.done = False
        self.actions = []

    def observe(self):
        pil_obs = Image.fromarray(self.cur_obs, 'RGB')
        pil_obs = pil_obs.resize((64,64))
        return pil_obs #uint

    def true_observe(self):
        return self.cur_obs #uint

    def to_tensor(self, obs):
        return (to_tensor_func(obs)).unsqueeze(0)

    def step(self, a):
        obs0 = self.observe()
        
        observation, reward, done, info = self.env.step(a)
        
        self.cur_obs = observation
        self.done = done

        assert isinstance(reward, float) or isinstance(reward, int), f"Reward is not float, but {type(reward)}: {reward}"
        return self.observe(), reward, done, info #played

    def render(self):
        self.observe().show()

    def reset(self):
        self.cur_obs = (self.env.reset())
        return self.observe()

