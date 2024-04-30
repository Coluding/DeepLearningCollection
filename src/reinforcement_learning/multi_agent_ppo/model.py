import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta, Categorical


class ContinousActorNetwork(nn.Module):
    """
    TLDR:
    This model is not a deterministic policy network, but a stochastic policy network. Even though it outputs a single
    action, it is actually a distribution of actions. The output is a Beta distribution, which is a distribution of
    probabilities. The Beta distribution is a continuous distribution, which is why it is used in this model. We are doing
    this since the PPO algorithm is a continuous action space algorithm. The Beta distribution is a continuous distribution

    Detailed:
    Implements the Proximal Policy Optimization (PPO) actor model for reinforcement learning in environments with
    continuous action spaces, utilizing stochastic policies.

    PPO is a policy gradient method designed to address the challenges of sample efficiency, stability, and reliable
    learning in deep reinforcement learning. By employing stochastic policies, PPO facilitates effective exploration
    and manages the complexity inherent in continuous action spaces. The algorithm's core features include:

    - Sample Efficiency: Optimizes the policy with an objective that encourages judicious updates, making efficient use
    of collected samples by limiting the magnitude of policy changes, thus maximizing learning from each interaction with the environment.

    - Stability and Reliability: Incorporates a clipping mechanism to prevent excessively large policy updates, ensuring
     gradual and stable learning progression. This feature is crucial in continuous action spaces where small adjustment
     can significantly impact performance.

    - Flexibility and Simplicity: PPO's straightforward implementation and its ability to perform well across a diverse
     set of environments make it a versatile tool for various applications, from robotics to game playing.

    - Exploration: The use of stochastic policies inherently promotes exploration by sampling actions from a probability
     distribution. This approach, coupled with entropy bonuses, helps the agent to explore the action space more
      thoroughly, aiding in the discovery of optimal strategies.

    PPO's implementation for continuous action spaces typically involves a policy network that outputs parameters of a
     distribution (e.g., Gaussian) from which actions are sampled. This design allows for both effective exploration
     and the nuanced control needed in continuous environments.

    Usage:
    Designed for use in environments with continuous action spaces where the agent's objective is to learn optimal
    or near-optimal policies through interaction and iterative improvement. PPO's combination of efficiency, stability,
     and exploration makes it particularly well-suited for complex tasks requiring continuous control decisions.

    """

    def __init__(self, n_actions, input_dims, alpha, fc1_dims=128, fc2_dims=128,
                 chkpt_dir='tmp/continous_actor', scenario=None):
        super().__init__()
        chkpt_dir += scenario
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'actor_continuous_ppo')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.alpha = nn.Linear(fc2_dims, n_actions)
        self.beta = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        The model outputs alpha and beta params of the Beta distribution. This is done to output a continuous value for
        each action between 0 and 1. The Beta distribution is a continuous distribution, which is why it is used in
        this model.

        Dosctring:
        Forward pass to predict the parameters of the Beta distribution for each action dimension.
        This model is designed for environments where actions are continuous and bounded within the [0, 1] interval.
        It outputs the alpha and beta parameters of the Beta distribution, which are used to model the policy's action
        selection mechanism. The Beta distribution is particularly suited for such tasks due to its flexibility in
        modeling behaviors ranging from deterministic to highly uncertain, all within the bounded [0, 1] range.

        :param self: The object instance.
        :param state: The input state tensor.
        :return: The Beta distribution with the predicted alpha and beta parameters.
        """
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        dist = Beta(alpha, beta)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,
                 fc1_dims=128, fc2_dims=128, chkpt_dir='models/',
                 scenario=None):
        super(ContinuousCriticNetwork, self).__init__()
        chkpt_dir += scenario
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'critic_continuous_ppo')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
