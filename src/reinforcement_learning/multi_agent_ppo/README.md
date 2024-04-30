https://github.com/philtabor/Multi-Agent-Reinforcement-Learning/tree/main/mappo/mappo

## The Probability Ratio in PPO and How We Implement It

In Proximal Policy Optimization (PPO), a core concept is the probability ratio, which plays a crucial role in stabilizing policy updates. Understanding and implementing this ratio correctly is essential for the effectiveness of the PPO algorithm. Initially, there was confusion about summing log probabilities across action dimensions, which is a critical step in calculating this ratio for multi-dimensional or sequential actions.

### Problem in Understanding

The confusion stemmed from the operation of summing log probabilities. It was unclear why and how this operation was necessary, especially when considering the joint probability of multi-dimensional actions under a policy.

### How We Solved It

We clarified the process with the following explanation, which became a guiding principle in our implementation:

Summing log probabilities across action dimensions to compute the joint probability of selecting a specific action vector.
This method leverages the logarithmic identity log(a*b) = log(a) + log(b), ensuring numerical stability and efficiency.
It effectively calculates the likelihood of the entire action sequence under the policy, assuming independent action components.


This comment helped demystify the process, highlighting the importance of summing log probabilities to compute the joint probability of multi-dimensional actions efficiently and accurately. It underscores the logarithmic identity that makes this operation equivalent to multiplying probabilities, ensuring numerical stability in our calculations.

See the agent.learn() method in the code snippet below for a practical example of how we implemented this concept in our PPO algorithm:

```python
old_probs = old_probs_arr[batch]
actions = action_arr[batch]
actor_state = actor_states_arr[batch]
dist = self.actor(actor_state)           
# use log prob summing trick for numerical stability
prob_ratio = T.exp(new_probs.sum(2, keepdims=True) - old_probs.sum(2, keepdim=True))
```
