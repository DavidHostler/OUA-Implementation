# Ornstein Uhlenbeck Adaptive Process

This is my implementation from scratch of the paper Ornstein Uhlenbeck Adaptation as a Mechanism for Learning 
In Brains and Machines (Jesus Garcia Fernandez et al, Radboud University, Netherlands).

Link here:

https://arxiv.org/abs/2410.13563

In this implementation, we model the usage of the OUA mechanism to train a multiparameter model to approximate 
simple datasets. 

## OUA 

Ornstein Uhlenbeck noise is a common feature traditionally used in Reinforcement Learning (RL) algorithms, 
most notably Deep Deterministic Policy Gradient (DDPG) methods. With the advent of more sample efficient techniques 
such as Soft-Actor Critic approaches and John Schulmann's Proximal Policy Optimization algorithm (PPO, made famous 
by ChatGPT), OU has fallen out of common discussion.

The research team behind this paper have introduced it for much the same reason as is used in RL- to introduce 
exploration of the action/state space via noise. 
Equations (1), (2), and (5) introduce noise in the form of Stochastic Differential Equations, which almost never 
arise explicitly in machine learning outside of quantitative finance. 
Equation (5) in particular is touted as being designed to facilitate a type of 'mean reversion', i.e. sampling 
updates to model weights in the direction of some mean parameter value, "mu". 
Comparing equation (5) with the general form for the Stochastic Differential Equation from Langevin Dynamics suggests 
that the ideal parameter distribution is a multivariate Gaussian with mean "mu", and a standard deviation given by 
1 / lambda_.

dθ(t)=λ(μ(t)−θ(t))dt+ΣdW(t)

The mean in this equation is calculated by solving an Ordinary Differential Equation and is a direct function of a solution 
to yet another ODE, the Reward Prediction Error (RPE) equation. 
Each set of equations is solved numerically and hierarchically, so updates flow from the errors in the output inference 
prediction downstream toward solving the mean ODE and then the parameter SDE directly. 

The original implementation provided by the authors uses Python libraries to handle the numerical methods for integration- 
in order for me to learn, I have done no such thing. Euler-Maruyama integration is integrated by hand, as it is very 
straightforward. 

## Implications 


### Reinforcement Learning 
The OUA mechanism based off of neurobiological systems within the eponymous paper appears to carry out parameters 
updates by Langevin sampling toward a region of lowest energy (the mean) characterized by high reward values. 
Langevin Dynamics as used herein are identical to the methods underlying Diffusion Models, however we are sampling 
over a distribution of model parameters as opposed to high-dimensional vector representations of some underlying 
image dataset.

Multivariate Gaussians are high-dimensional, and some of my earlier work indicates that even though these distributions 
are well-defined, it is difficult to use RL via a smaller model to even learn a loss/reward landscape over the primary 
model's parameter space. Diffusive learning via either Langevin Dynamics or Gibs Sampling promises a more efficient 
approach, especially when the target distribution is well-defined. 

This is one of the nice aspects of RL- it often allows us to pick or 'shape' our requisite reward functions, as opposed 
to performing gradient descent along some continuous loss function whose maxima and minima remain a total mystery to the 
machine learning engineer involved.

Traditional Supervised Learning will therefore not benefit significantly from this advantage, and therefore it can be 
reasoned that perhaps the future lies with some kind of dopaminergic RL-based learning.

## TL;DR 

Python implementation of a Dutch paper, showing that reinforcement learning can be carried out to model basic 
functions by iteratively solving a series of stochastic and ordinary differential equations.

