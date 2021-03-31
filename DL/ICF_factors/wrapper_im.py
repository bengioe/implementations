#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    disentangle_rl.wrapper
    ~~~~~~~~~~~~~~~~~~~~~~

    Wrapper to make MazeBase behave like openAI Gym

    :copyright: (c) 2017 by Valentin THOMAS.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice, randint
import ipdb
from time import sleep
from os import system
from pprint import PrettyPrinter
from six.moves import input
import numpy as np

import mazebase.games as games
from mazebase.games import featurizers
from mazebase.games import curriculum
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Height and width of the MazeBase environment
# ENV_HEIGHT = 2
# ENV_WIDTH = 2

# Size of the cells in the final plot (MazeBase images are 32 pixels)
CELL_SIZE = 32
water_block = np.asarray(mpimg.imread('images/water.png'))
agent_block = np.asarray(mpimg.imread('images/agent.png'))
block_block = np.asarray(mpimg.imread('images/block.png'))
red = np.array([1, 0, 0])
red_pos = ((agent_block == red).sum(axis=2) == 3)

class Number:
    def __init__(self, n):
        self.n = n

class Env:

    """Docstring for Env. """

    def __init__(self, size, n_switches, light=0):
        """TODO: to be defined1. """
        self.attributes = [0]*(2+n_switches)
        self.ENV_WIDTH = size
        self.ENV_HEIGHT = size
        switches = curriculum.CurriculumWrappedGame(
            games.Switches,
            waterpct=0.0, blockpct=0.0, n_switches=n_switches,
            switch_states=2,
            curriculums={
                'map_size': games.curriculum.MapSizeCurriculum(
                    (self.ENV_WIDTH, self.ENV_WIDTH, self.ENV_HEIGHT, self.ENV_HEIGHT),
                    (1, 1, 1, 1),
                    (10, 10, 10, 10)
                )
            }
        )
        self.light=light
        self.light_state = 1
        # mg = curriculum.CurriculumWrappedGame(
        #     games.MultiGoals,
        #     waterpct=0.3,
        #     curriculums={
        #         'map_size': games.curriculum.MapSizeCurriculum(
        #             (3, 3, 3, 3),
        #             (3, 3, 3, 3),
        #             (10, 10, 10, 10)
        #         )
        #     }
        # )

        game = games.MazeGame(
             [switches],
            featurizer=featurizers.GridFeaturizer()
        )
        self.game = game
        self.actions = game.all_possible_actions()
        self.actions = game.actions()
        self.action_space = Number(len(self.actions))
        self.observation_space = Number(int(np.prod(self.observe().shape)))


    def observe(self):
        observation_matrix = np.asarray(self.game.observe()['observation'])[0,: :]
        image = 1 * np.ones((CELL_SIZE*self.ENV_HEIGHT, CELL_SIZE*self.ENV_WIDTH, 3))

        # For each cell in the environment
        #   Parse all objects and add the object image to the output
        sum_states = 0
        switch_count = 0
        for x in range(self.ENV_WIDTH):
            for y in range(self.ENV_HEIGHT):
                is_switch=False
                is_agent=False
                is_block=False
                is_water=False
                for object in observation_matrix[x,y]:
                    cell = np.ones((CELL_SIZE, CELL_SIZE, 3))
                    if object[:5] == 'state':
                        switch_state = int(object[5])
                        is_switch=True
                        self.attributes[2+switch_count] = switch_state
                        switch_count+=1
                        sum_states += switch_state
                    elif object == 'Agent':
                        is_agent=True
                        self.attributes[0] = y
                        self.attributes[1] = x
                    elif object == 'Water':
                        is_water=True
                    elif object=='Block':
                        is_block=True

                cell = np.ones((CELL_SIZE, CELL_SIZE, 3))
                for object in observation_matrix[x,y]:

                    # We can add all the other objects easily copying these lines
                    if object[:5] == 'state':
                        switch_state = int(object[5])
                        if switch_state % 2 == 0:
                            green_cell = np.zeros((CELL_SIZE, CELL_SIZE, 3))
                            green_cell[:,:] = np.asarray([0, 1, 0])
                            cell = green_cell# / (1+switch_state)
                        else:
                            blue_cell = np.zeros((CELL_SIZE, CELL_SIZE, 3))
                            blue_cell[:,:] = np.asarray([0, 0, 1])
                            cell = blue_cell

                    # elif object == 'Agent':
                    #     ag = np.asarray(mpimg.imread('images/agent.png'))

                    elif object == 'Water':
                        cell = np.copy(water_block)

                    # elif object == 'Switch':
                        # object_value = np.asarray([-1, -1, -1])


                    elif object == 'Block':
                        cell = np.copy(block_block)

                    if not (cell == 1).all():   # If there is a plottable object
                        image[(CELL_SIZE*y):(CELL_SIZE*(y+1)), (CELL_SIZE*x):(CELL_SIZE*(x+1)), :] = cell

                    # If the Agent is in the current cell it should overwrite any other object

                    #if object == 'Agent':
                    #    break
                if is_agent:
                    cell[red_pos] = np.copy(agent_block[red_pos])
                    image[(CELL_SIZE*y):(CELL_SIZE*(y+1)), (CELL_SIZE*x):(CELL_SIZE*(x+1)), :] = cell

        self.light_state = (sum_states % 2)

        if self.light == 0:
            return image
        else:
            return image/(1+self.light_state)

    def step(self, a):
        obs0 = self.observe()
        self.game.act(self.actions[a])
        r = self.game.reward()
        done = self.game.is_over()
        obs = self.observe()
        if np.allclose(obs0, obs):
            played = False
        else:
            played = True

        assert isinstance(r, float) or isinstance(r, int), f"Reward is not float, but {type(r)}: {r}"
        return obs, r, done, played

        self.game.display()
        plt.imshow(self.observe(), origin='lower')
        plt.show(block=False)
        plt.pause(0.5)

    def render_in_console(self):
        self.game.display()

    def old_observe(self):
        config = self.game.observe()
        #pp.pprint(config['observation'][1])
        obs, info = config['observation']
        featurizers.grid_one_hot(self.game, obs)
        obs = np.array(obs)
        # obs is now a 10x10xN 1-0 tensor
        featurizers.vocabify(self.game, info)
        # info is now a 10x10 integer tensor
        config['observation'] = obs, info
        ## ================== ##
        tot_obs = np.concatenate((obs.ravel(), np.array(info).ravel()))
        #return tot_obs
        return obs.ravel()-0.1

    def reset(self):
        self.game.reset()
        # VERY IMPORTANT:
        # otherwise returns NaN as the current state
        #self.game.game = self.game.games[0]
        # while self.game.is_over():
        #     self.game.reset()
        return self.observe()

