import collections
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import gym
import cv2 as cv
from gym.core import ObsType


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        """
        This wrapper is used to repeat the action and return the maximum frame in order to handle the flickering issue in
        atari games.
        :param env: Environment
        :param repeat: The number of times to repeat the action
        """
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.shape
        self.frame_buffer = np.zeros((2, *self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):

        t_reward = 0
        done = False

        for i in range(self.repeat):
            obs, reward, done, truncated,  info = self.env.step(action)

            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs

            if done:
                break
        # we do this to remove flickering. This possibly leads to the case that the agent akes action a but the result
        # is the state s4 which is the result of taking the action 4 times in state s0. However, this tradeoff ensures
        # the input data is not noisy.
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, truncated, info

    def reset(self):
        # Reset the environment to its initial state and get the initial observation.
        obs = self.env.reset()

        # Determine the number of no-op (no operation) actions to perform at the start.
        # This introduces variability in the starting states of each episode.
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0

        # Perform the determined number of no-op actions.
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)  # Action '0' typically represents a no-op.
            # If a no-op action ends the episode (unlikely but possible in some environments),
            # reset the environment again. This ensures the episode doesn't start in a terminal state.
            if done:
                self.env.reset()

        # If the 'fire first' flag is true, perform an initial 'fire' action,
        # typically required to start or resume many games.
        if self.fire_first:
            # Assert that the action corresponding to 'FIRE' is indeed intended to start/resume the game.
            # This is a sanity check to ensure the environment's action meanings align with expectations.
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            # Perform the 'fire' action and update the initial observation.
            obs, _, _, _ = self.env.step(1)

        # Initialize the frame buffer to store the current and previous observations.
        # This is relevant for wrappers that process or compare consecutive frames.
        self.frame_buffer = np.zeros((2, *self.shape))
        # Store the current observation as the initial state in the frame buffer.
        if not isinstance(obs, np.ndarray):
            obs, info  = obs
        self.frame_buffer[0] = obs

        # Return the initial observation to start the episode.
        return obs, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, obs):
        if obs.dtype != np.uint8:
            obs = obs.astype(np.uint8)
        new_frame = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
        resized_screen = cv.resize(new_frame, self.shape[1:], interpolation=cv.INTER_AREA)

        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0), dtype=np.float32,
                                                )

        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation, _ = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(env_name, render_mode='rgb_array')
    env = RepeatActionAndMaxFrame(env, repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env