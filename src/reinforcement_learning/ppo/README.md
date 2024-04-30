This code was used from and inspired by:
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/


## General Advantage Estimation
In comparison to the REINFORCE algorithm, PPO uses the General Advantage Estimation (GAE) to estimate the advantage function. The GAE is a method to estimate the advantage function by using the value function to reduce the variance of the advantage function. 
Instead of simply using the TD error as the advantage function, the GAE uses the following formula to estimate the advantage function:
\[

A(s_t, a_t) = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots + (\gamma \lambda)^{T-t+1} \delta_{T-1}

\]
where \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\) is the TD error, \(\lambda\) is the GAE parameter, and \(T\) is the length of the trajectory. The GAE parameter \(\lambda\) is a hyperparameter that determines how much the value function should be used to estimate the advantage function. A value of \(\lambda = 0\) corresponds to the TD error, while a value of \(\lambda = 1\) corresponds to the Monte Carlo estimate of the advantage function. In practice, a value of \(\lambda = 0.95\) is commonly used.
