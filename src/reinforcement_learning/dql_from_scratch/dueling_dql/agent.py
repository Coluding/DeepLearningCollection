import numpy as np
import torch
from dql_from_scratch.dueling_dql.model import DuelingDeepQNetwork
from dql_from_scratch.atari.agent import DQNAgent
from dql_from_scratch.atari.replay_memory import ReplayBuffer


class DuelingDQNAgent(DQNAgent):

    def __init__(self, gamma, eps, eps_min, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_dec=1e-5, replace=1000,
                 algo=None, env_name=None,
                 chkpt_dir='tmp/dueling_dqn'):
            super(DuelingDQNAgent, self).__init__(gamma, eps, eps_min, lr, n_actions, input_dims,
                  mem_size, batch_size, eps_dec, replace, algo, env_name, chkpt_dir)

            self.q_eval = DuelingDeepQNetwork(self.lr,
                                              self.n_actions,
                                              input_dims=input_dims,
                                              name=self.env_name+'_'+self.algo+'_q_eval',
                                              chp_dir=self.chkpt_dir)

            self.q_target = DuelingDeepQNetwork(self.lr,
                                                self.n_actions,
                                                input_dims=input_dims,
                                                name=self.env_name+'_'+self.algo+'_q_target',
                                                chp_dir=self.chkpt_dir)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_target(states_)

        indices = np.arange(self.batch_size)

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)).detach()).max(dim=1)[0]

        q_next[dones.to(bool)] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


    def choose_action(self, observation, deterministic=False):
        if np.random.random() < self.eps and not deterministic:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            V, A = self.q_eval(state)
            action = torch.argmax(A).item()

        return action


class DuelingDDQNAgent(DuelingDQNAgent):
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_target(states_)

        indices = np.arange(self.batch_size)

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)).detach()).max(dim=1)[0]
        suggested_actions = torch.argmax(A_s_, dim=1)

        q_next[dones.to(bool)] = 0.0
        q_target = rewards + self.gamma * q_next[indices, suggested_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

