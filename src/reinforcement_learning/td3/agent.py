import torch
import numpy as np

from model import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer


class TD3Agent:
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, warmup=1000,
            n_actions=2, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions, name='target_critic_2')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, state, inference=False):
        if inference:
            state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            return mu.cpu().detach().numpy()

        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)

        else:
            state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)


        mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise),
                                   dtype=torch.float).to(self.actor.device)

        mu_prime = torch.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """
            Executes a learning step for the Twin Delayed DDPG (TD3) agent.

            This method updates both the Critic and Actor networks based on a batch of
            experiences sampled from the replay buffer. It follows the TD3 algorithm's
            key steps: updating the Critic networks using the minimum of the twin
            Critic's Q-values to mitigate overestimation bias, and delayed updating of
            the Actor network to ensure policy updates are based on more stable Q-value
            estimates. Target networks are softly updated to further enhance stability.

            The method encompasses the following steps:
            1. Sampling a batch of experiences from the replay buffer.
            2. Computing the target actions using the target Actor network, with added
               noise for exploration and clipping to ensure actions remain valid.
            3. Calculating the next-state Q-values using the target Critic networks,
               taking the minimum of the two to mitigate overestimation.
            4. Computing the target Q-values, considering done flags to handle episode
               ends.
            5. Updating the Critic networks by minimizing the MSE loss between the
               current Q-values and the target Q-values.
            6. Every `update_actor_iter` steps, updating the Actor network by maximizing
               the Q-values estimated by one of the Critic networks.
            7. Softly updating the target networks to slowly track the learned networks.

            Preconditions:
            - The replay buffer must have at least `batch_size` experiences to sample
              from.
            - `learn_step_cntr` is used to track the number of learning steps and
              determine when to update the Actor network.

            Updates:
            - Critic networks' parameters are updated every call.
            - Actor network's parameters are updated every `update_actor_iter` calls.
            - Target networks are softly updated every call after updating the Actor and
              Critic networks.

            Note:
            - Target actions are perturbed with noise and clipped to ensure the exploration
              remains effective and the actions are within the environment's valid range.
            - Done flags are used to zero out the Q-values for the final states, preventing
              incorrect value propagation.
            """
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
        done = torch.tensor(done).to(self.critic_1.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic_1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)

        target_actions = torch.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        # This is the core of the TD3 algorithm
        critic_value_ = torch.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = torch.nn.MSELoss()(target, q1)
        q2_loss = torch.nn.MSELoss()(target, q2)

        critic_loss = q1_loss + q2_loss

        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()

        # We are always using the critic_1 to update the actor
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = - torch.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                              (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau * critic_2[name].clone() + \
                             (1 - tau) * target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau * actor[name].clone() + \
                          (1 - tau) * target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self, path=None):
        self.actor.load_checkpoint(path)
        self.target_actor.load_checkpoint(path)
        self.critic_1.load_checkpoint(path)
        self.critic_2.load_checkpoint(path)
        self.target_critic_1.load_checkpoint(path)
        self.target_critic_2.load_checkpoint(path)
