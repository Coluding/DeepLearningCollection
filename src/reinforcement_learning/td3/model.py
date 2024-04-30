import torch.nn as nn
import torch
import os


class CriticNetwork(nn.Module):

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="tmp/td3_backup", *args, **kwargs):

        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(torch.cat([state, action], dim=1))
        q1_action_value = torch.relu(q1_action_value)
        q1_action_value = torch.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path=None):
        print('... loading checkpoint ...')
        if path is not None:
            self.load_state_dict(torch.load(os.path.join(path, self.name + '_td3')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))



class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
            n_actions, name, chkpt_dir='tmp/td3_backup'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        """
        Action space ranges from -1 to 1, so we use tanh activation function to scale the output to this range.
        :param state:
        :return:
        """
        prob = self.fc1(state)
        prob = torch.relu(prob)
        prob = self.fc2(prob)
        prob = torch.relu(prob)

        mu = torch.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path=None):
        print('... loading checkpoint ...')
        if path is not None:
            self.load_state_dict(torch.load(os.path.join(path,self.name + '_td3')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))

