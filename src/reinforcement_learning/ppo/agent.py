import torch as T
import numpy as np

from memory import PPOMemory
from model import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(alpha, input_dims, n_actions)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)

        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value


    def learn(self, individual_update=False):
        """
            Performs policy and value network updates using the Proximal Policy Optimization (PPO) algorithm,
            leveraging Generalized Advantage Estimation (GAE) for advantage calculation.

            The method iterates over epochs, each consisting of processing minibatches of collected experiences.
            Experiences are composed of states, actions, rewards, and the terminal status of episodes, along with
            the log probabilities of actions under the policy at the time of experience collection and the value
            estimates for each state.

            Advantage Calculation:
            - Unlike simple TD(0) error that estimates the advantage based on the immediate next state, GAE
              accumulates discounted future TD errors to estimate the advantage of each state-action pair. This
              accumulation considers not just the immediate reward and next state's value but extends to multiple
              future states, weighted by a decay factor combining the discount rate (gamma) and the GAE parameter
              (lambda).
            - The inner loop computes these accumulated discounted TD errors for each timestep, effectively
              capturing the long-term consequences of actions beyond the next immediate step. This provides a more
              nuanced and balanced estimate of the advantage, helping to inform more effective policy updates.

            Policy (Actor) and Value (Critic) Updates:
            - The policy network is updated to increase the probability of actions that lead to higher than expected
              returns, as indicated by the calculated advantages. The updates are regulated by a clipping mechanism
              to prevent excessive deviations from the previous policy, promoting stable learning.
            - The value network is updated to minimize the discrepancy between its estimates and the computed
              returns, aiming to improve its predictions of future returns from each state.

            The function encapsulates the core of the PPO learning process, including the calculation of GAE for
            advantage estimation, and the subsequent updates to the policy and value networks based on these
            advantages. The goal is to iteratively improve the policy towards achieving higher returns by
            efficiently exploring the action space and accurately estimating state values.

            Parameters:
            None explicitly passed; relies on data stored in the object's memory attribute from prior
            interactions with the environment.

            Returns:
            None; updates the actor (policy) and critic (value) networks in-place.
            """

        # TODO: check whether shuffling corrupts the sequential nature of the data, which conflicts with the GAE calculation
        # in the generate_batches method
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

        values = vals_arr

        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        # compute GAE advantages (shuffling may corrupt this)
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage).to(self.actor.device)

        values = T.tensor(values).to(self.actor.device)

        for batch in batches:
            states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
            old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
            actions = T.tensor(action_arr[batch]).to(self.actor.device)
            dist = self.actor(states)
            critic_value = self.critic(states).squeeze()

            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()

            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

            returns = advantage[batch] + values[batch]
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()

            # update actor and critic networks aligning with the PPO algorithm
            # we update the actor and the critic separately
            if individual_update:
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

            else:
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()



    def run_learning(self):
        for i in range(self.n_epochs):
            self.learn()
            self.memory.clear_memory()


    def play(self, env, n_games=1):
        observation = env.reset()
        if not isinstance(observation, np.ndarray):
            observation = observation[0]
        done = False
        score = 0
        for _ in range(n_games):
            while not done:
                action, _, _ = self.choose_action(observation)
                observation_, reward, done, info, _ = env.step(action)
                if not isinstance(observation_, np.ndarray):
                    observation_ = observation_[0]
                score += reward
                observation = observation_
            return score


