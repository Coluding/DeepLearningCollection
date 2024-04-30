import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from agent import TD3Agent
from utils import plot_learning_curve

def main():
    play = True
    env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array' if play else None)
    agent = TD3Agent(alpha=0.0001, beta=0.0001,
            input_dims=env.observation_space.shape, tau=0.01,
            env=env, batch_size=256, layer1_size=512, layer2_size=512,
            n_actions=env.action_space.shape[0])


    if play:
        before_training = "trained_lunar.mp4"
        video = VideoRecorder(env, before_training)
    n_games = 20
    filename = 'plots/td3_' + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []

    agent.load_models("/home/lubi/Documents/Projects/RL/rl/td3/tmp/td3")

    for i in range(n_games):
        observation = env.reset()
        if not isinstance(observation, np.ndarray):
            observation = observation[0]

        done = False
        score = 0

        while not done:
            if play:
                env.render()
            action = agent.choose_action(observation, inference=play)
            observation_, reward, done, trunc, info = env.step(action)
            if play:
                video.capture_frame()

            else:
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()

            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if not play:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()


        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if play:
        video.close()
        return
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, filename)


if __name__ == '__main__':
    main()