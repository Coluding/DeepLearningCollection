import numpy as np
import torch as T
from model import ContinousActorNetwork, ContinuousCriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims,
                 n_actions, agent_idx, agent_name,
                 gamma=0.99, alpha=3e-4, T=2048,
                 gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10,
                 n_procs=8, chkpt_dir=None,
                 scenario=""):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.n_procs = n_procs

        self.actor = ContinousActorNetwork(n_actions, actor_dims, alpha,
                                          chkpt_dir=chkpt_dir,
                                          scenario=scenario)
        self.critic = ContinuousCriticNetwork(critic_dims, alpha,
                                              chkpt_dir=chkpt_dir,
                                              scenario=scenario)
        self.n_actions = n_actions

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)

            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)

        return action.cpu().numpy(), probs.cpu().numpy()

    def calc_adv_and_returns(self, memories):
        """
        This computes advantages based on the Generalized Advantage Estimation (GAE) algorithm.
        We loop through the memories in reverse order to calculate the advantages and scaling them with lambda and gamma.
        If lambda equals 0, the advantage is the TD error. If lambda equals 1, the advantage is the sum of all future rewards,
        i.e. representing Monte Carlo estimation.
        :param memories:
        :return:
        """
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states).squeeze()
            values_ = self.critic(new_states).squeeze()
            deltas =r[:,:, self.agent_idx] + self.gamma * values_ * (1 - dones) - values
            deltas = deltas.cpu().numpy()
            adv = [0]

            # compute GAE
            # the advantage at time step t is the discounted sum of future advantages plus the current advantage
            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] + self.gamma * self.gae_lambda * adv[-1] * dones[step].cpu().numpy()
                adv.append(advantage)

            adv.reverse()
            adv = np.array(adv[:-1])
            adv = T.tensor(adv, device=self.critic.device).unsqueeze(2)

            # look how the delta was computed --> we are adding the value back so we get the returns
            returns = adv + values.unsqueeze(2)
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        return adv, returns


    def learn(self, memory):
        actor_states, states, actions, old_probs, rewards, actor_new_states, \
            states_, dones = memory.recall()
        device = self.critic.device
        states_arr = T.tensor(states, dtype=T.float, device=device)
        states__arr = T.tensor(states_, dtype=T.float, device=device)
        r = T.tensor(rewards, dtype=T.float, device=device)
        action_arr = T.tensor(actions[self.agent_name],
                              dtype=T.float, device=device)
        old_probs_arr = T.tensor(old_probs[self.agent_name], dtype=T.float,
                                 device=device)
        actor_states_arr = T.tensor(actor_states[self.agent_name],
                                    dtype=T.float, device=device)

        dones_arr = T.tensor(dones, dtype=T.float, device=device)
        adv, returns = self.calc_adv_and_returns((state_arr, states__arr,
                                                  r, dones_arr))


        # we are training with the same trajectory for all agents over n epochs
        # afterwards we collect new trajectories and train again
        for epoch in range(self.n_epochs):
            batches = memory.generate_batches()

            for batch in batches:
                old_probs = old_probs_arr[batch]
                actions = action_arr[batch]
                actor_state = actor_states_arr[batch]
                dist = self.actor(actor_state)
                # we sample actions from the old policy and plug them into our distribution to see
                # what the new probabilities under this distribution are
                new_probs = dist.log_prob(actions)

                # we currently have the log probs for each continous action. We want to get the prob ratio
                # therefore, we need to sum over the log probs, which essentially means we multiply the probs using the
                # log rule for multiplication. hence, by summing we get the log prob seeing the actions under the new policy
                # If we divide this by the log prob of the old policy, we get the ratio of the new probs to the old probs
                # We sure have to take the exponential first to get the actual probabilities
                prob_ratio = T.exp(new_probs.sum(2, keepdims=True) - old_probs.sum(2, keepdim=True))

                # Summing log probabilities across action dimensions to compute the joint probability of selecting a specific action vector.
                # This method leverages the logarithmic identity log(a*b) = log(a) + log(b), ensuring numerical stability and efficiency.
                # It effectively calculates the likelihood of the entire action sequence under the policy, assuming independent action components.

                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv[batch]
                entropy = dist.entropy().sum(2, keepdims=True)

                # This is the PPO optimization goal: Maxaimize the advantage w.r.t. to the policy parameters
                actor_loss = -T.min(
                    weighted_probs,
                    weighted_clipped_probs
                ) - entropy * self.entropy_coefficient
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm(self.actor.parameters(), 30)
                self.actor.optimizer.step()

                states = states_arr[batch]
                critic_value = self.critic(states).squeeze()

                critic_loss = T.nn.MSELoss()(returns[batch], critic_value)
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                T.nn.utils.clip_grad_norm(self.critic.parameters(), 30)
                self.critic.optimizer.step()




