import numpy as np
import torch
from dql_from_scratch.atari.model import DeepQNetwork
from dql_from_scratch.atari.agent import DQNAgent
from dql_from_scratch.atari.replay_memory import ReplayBuffer


class DDQNAgent(DQNAgent):
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]
        suggested_action = torch.argmax(self.q_eval(states_).detach(), dim=-1)
        q_next = self.q_target(states_).detach()[indices, suggested_action]
        q_target = rewards + self.gamma * q_next
        q_next[dones.to(bool)] = 0.0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()





