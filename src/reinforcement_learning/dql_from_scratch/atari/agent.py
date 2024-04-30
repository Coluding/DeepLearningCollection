import numpy as np
import torch
from dql_from_scratch.atari.model import DeepQNetwork
from replay_memory import ReplayBuffer


class DQNAgent:
    def __init__(self, gamma, eps, eps_min, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_dec=1e-5, replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=input_dims,
                                   name=self.env_name+'_'+self.algo+'_q_eval', chp_dir=self.chkpt_dir)

        self.q_target = DeepQNetwork(self.lr, self.n_actions, input_dims=input_dims,
                                     name=self.env_name+'_'+self.algo+'_q_target', chp_dir=self.chkpt_dir)

    def choose_action(self, observation, deterministic=False):
        if np.random.random() < self.eps and not deterministic:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        states_ = torch.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_target.load_checkpoint()

    def decrement_epsilon(self):
        self.eps = self.eps - self.eps_dec \
            if self.eps > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = torch.max(self.q_target.forward(states_).detach(), dim=1)[0]
        q_target = rewards + self.gamma * q_next
        q_next[dones.to(bool)] = 0.0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()





