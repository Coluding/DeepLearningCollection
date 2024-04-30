import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder


from agent import DQNAgent
from dql_from_scratch.double_q_learning.agent import DDQNAgent
from dql_from_scratch.dueling_dql.agent import DuelingDQNAgent
from utils import *


def main():
    env = make_env('ALE/Breakout-v5')
    best_score = -np.inf
    load_checkpoint = False
    play = False
    if play:
        before_training = "trained_pong.mp4"
        video = VideoRecorder(env, before_training)
    n_games = 2000
    agent = DuelingDQNAgent(gamma=0.99, eps=1.0, lr=0.0001,
                     input_dims=env.observation_space.shape,
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DuelDQNAgent',
                     env_name='Breakout-v5')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'

    figure_file = 'plots/' + fname + '.png'


    n_steps = 0

    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False

        observation = env.reset()
        score = 0

        while not done:
            env.render()
            if play:
                video.capture_frame()
            action = agent.choose_action(observation, True)
            observation_, reward, done, _, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.eps, 'steps', n_steps)
        if play:
            video.close()
            return
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.eps)

    x = [i - 1 for i in range (1, n_games + 1)]

    plotLearning(x, scores, eps_history, figure_file)


if __name__ == '__main__':
    main()