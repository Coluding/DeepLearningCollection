import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.actor_state_memory = None
        self.actor_new_state_memory = None
        self.actor_action_memory = None

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1


    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actor_actions = []

        for i in range(self.n_agents):
            actor_states.append(self.actor_state_memory[i][batch])
            actor_new_states.append(self.actor_new_state_memory[i][batch])
            actor_actions.append(self.actor_action_memory[i][batch])

        return states, actor_states, actor_actions, rewards, new_states, actor_new_states, dones


    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False

