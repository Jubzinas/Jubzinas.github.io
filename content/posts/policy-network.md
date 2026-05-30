---
title: "Policy Networks: Learning to Act Directly"
date: 2026-05-30
draft: false
tags: ["AI", "RL", "Policy Gradient", "REINFORCE"]
---

Unlike Q-Learning, which learns the value of actions and then derives a policy,
a **policy network** learns the policy directly. In other words, instead of asking
*“how good is this action in this state?”*, it asks *“what action should I take
right now?”*

That makes policy gradients especially useful when the action space is small but
the optimal behavior depends on long-term consequences that are hard to encode
with a table of values.

In this post I’ll walk through a simple **REINFORCE** agent trained on
**Acrobot-v1**, based on [this repository](https://github.com/Jubzinas/Policy-Network).

---

## Policy-Based Reinforcement Learning

Policy-based methods learn a function:

$$\pi_\theta(a \mid s)$$

which means “the probability of taking action $a$ in state $s$ given parameters
$\theta$.”

For Acrobot, the action space is discrete, so the policy network outputs a
probability for each possible action and then samples from that distribution.
The policy is therefore **stochastic**: the same state can lead to different
actions during training, which encourages exploration.

A simple policy network is usually:

1. An input layer for the environment state
2. One or more hidden layers
3. A final layer that outputs **action logits**
4. A **softmax** that turns logits into action probabilities

$$\pi_\theta(a \mid s) = \text{softmax}(f_\theta(s))$$

![The RL agent-environment loop](https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png)
*The standard RL loop: the agent observes a state, selects an action via its policy, receives a reward, and transitions to a new state. In policy-based methods the policy $\pi_\theta$ is the direct output of a neural network. ([Source: OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html))*

---

## Why Use a Policy Network?

Policy networks are useful when:

- you want to learn the action choice directly
- you need stochastic policies for exploration
- the environment is continuous or high-dimensional
- you care about optimizing long-term return rather than local action values

Instead of building a Q-table, the network learns parameters that map states
straight to action probabilities.

![RL Algorithm Taxonomy](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)
*A map of modern RL algorithms. REINFORCE belongs to the policy gradient family (right branch), which optimises the policy directly rather than estimating value functions first. ([Source: OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html))*

---

## REINFORCE: The Core Idea

REINFORCE is a **Monte Carlo policy gradient** method. The name comes directly
from the classic Williams (1992) paper — in practice we just think of it as:
run a full episode, measure how good it was, then push the policy parameters
in the direction that makes that outcome more likely.

### 1. The Objective

We want a policy $\pi_\theta$ that maximises the expected cumulative reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

where $\tau = (s_0, a_0, r_0, s_1, \ldots)$ is a full episode trajectory
sampled by following $\pi_\theta$, and $\gamma \in [0,1]$ is the discount factor.

### 2. The Policy Gradient Theorem

We need $\nabla_\theta J(\theta)$ to improve $\theta$, but the expectation is
over trajectories that themselves depend on $\theta$ — so we cannot just
differentiate the rewards directly.

The **Policy Gradient Theorem** (Sutton et al., 2000) solves this.
Using the **log-derivative trick**:

$$\nabla_\theta \pi_\theta(a \mid s) = \pi_\theta(a \mid s)\; \nabla_\theta \log \pi_\theta(a \mid s)$$

the gradient of the objective becomes an expectation we can estimate from
sampled episodes:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t \right]$$

where $G_t$ is the **discounted return from step $t$**:

$$G_t = \sum_{k=0}^{T-t} \gamma^k\, r_{t+k}$$

This is the total reward actually received from step $t$ onward, with future
rewards discounted — computed **after** the episode ends.

### 3. Intuition: What Does This Formula Say?

Break the gradient update into two parts:

| Term | Meaning |
|------|---------|
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | Direction in parameter space that makes action $a_t$ more likely in state $s_t$ |
| $G_t$ | How good was the outcome that followed? |

The product of the two gives:

- **Large $G_t$** — take a big step towards making $a_t$ more probable
- **Small or negative $G_t$** — take a small or reversed step, making $a_t$ less probable

This is exactly the trial-and-error logic: if what you did worked out,
do it more often; if it did not, do it less.

### 4. Why the Loss Is the Negative of the Objective

Neural network optimizers (SGD, Adam, ...) **minimize** a scalar loss.
Our objective $J(\theta)$ is something we want to **maximize**.

The standard trick is to flip the sign and define the loss as:

$$\mathcal{L}(\theta) = -\,\mathbb{E}\left[ \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t)\; G_t \right]$$

Minimizing $\mathcal{L}$ is mathematically identical to maximizing $J$:

$$\text{minimize } \mathcal{L} \iff \text{maximize } J$$

because $\nabla_\theta \mathcal{L} = -\nabla_\theta J$, so
gradient **descent** on $\mathcal{L}$ equals gradient **ascent** on $J$.

In code this looks like:

```python
# log_probs: list of log pi(a_t | s_t) collected during the episode
# returns:   list of G_t computed after the episode ends

loss = -torch.stack(
    [lp * G for lp, G in zip(log_probs, returns)]
).sum()

optimizer.zero_grad()
loss.backward()   # computes -gradient_J, i.e. the descent direction
optimizer.step()  # subtracts a small step -> effectively ascends J
```

The negative sign is the only thing that converts "gradient ascent on reward"
into "gradient descent on loss" — something every deep learning framework
already knows how to do.

### 5. Monte Carlo Sampling and Its Trade-offs

REINFORCE is a *Monte Carlo* method because $G_t$ is estimated from a
**complete sampled episode**, not from a bootstrapped value as TD-learning
or Actor-Critic methods do.

This gives an **unbiased** estimate of the true return, but with **high
variance**: two runs from the same initial state can yield very different
$G_t$ values because both the environment and the stochastic policy inject
randomness at every step.

Common mitigations are:

- **Baseline subtraction**: replace $G_t$ with $G_t - b(s_t)$ where $b$ is a
  learned state-value estimate. This keeps the gradient unbiased while
  reducing variance — the core idea behind Actor-Critic methods.
- **Return normalisation**: standardise the returns inside a batch to
  zero mean and unit variance before computing the loss.

For this project the raw return is used — keeping the implementation as
transparent as possible while still solving Acrobot.
---

## From Logits to Actions

The output of the network is not the action itself — it is a vector of logits.
Those logits are converted to probabilities using softmax:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

where $z_i$ is the logit for action $i$.

The agent then samples from that distribution. This sampling step is important:
if the network always picked the argmax action, it would explore much less.

---

## The Acrobot Environment

Acrobot is a classic control task from Gymnasium.
The agent controls the torque at the joint of a two-link pendulum and must swing
the end effector high enough to reach a target.

- **Environment**: `Acrobot-v1`
- **Action space**: 3 discrete torques
- **Goal**: swing the tip above the target height
- **Challenge**: rewards are sparse, so the agent must learn long-term strategy

This makes Acrobot a good testbed for policy gradients because success depends
on sequences of coordinated actions, not just immediate reward.

![Acrobot-v1 environment](https://gymnasium.farama.org/_images/acrobot.gif)
*The Acrobot-v1 environment: the agent applies torques to the middle joint and must swing the lower link above the horizontal line. Rewards are $-1$ per step until the goal is reached, making long-horizon planning essential. ([Source: Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/acrobot/))*

---

## Network Training Loop

The training loop in [policy_network.py](https://github.com/Jubzinas/Policy-Network/blob/main/policy_network.py) follows the standard REINFORCE pattern:

1. Run an episode using the current policy
2. Store log-probabilities of chosen actions
3. Compute discounted returns for the episode
4. Weight the log-probabilities by those returns
5. Backpropagate the loss and update the network

A simpler way to think about it:
- good trajectories push the policy up
- bad trajectories push the policy down

Because REINFORCE uses complete episode returns, the gradient estimate can be
noisy — but the method is conceptually simple and very effective for teaching
policy-gradient fundamentals.

---

## Before and After Training

The repository includes both the initial and learned behavior, plus a training
curve showing how the score improves over time.

### Start: untrained policy

![Acrobot before training](https://raw.githubusercontent.com/Jubzinas/Policy-Network/main/acrobot_start.gif)

An untrained policy behaves almost randomly and usually fails to swing the arm
high enough.

### End: trained policy

![Acrobot after training](https://raw.githubusercontent.com/Jubzinas/Policy-Network/main/acrobot_end.gif)

After training, the policy learns coordinated movements that generate the swing
needed to reach the target.

> All visuals above come directly from the [Policy-Network repository](https://github.com/Jubzinas/Policy-Network).

---

## Project Notes

The repository is organized into a small training/evaluation workflow:

- [policy_network.py](https://github.com/Jubzinas/Policy-Network/blob/main/policy_network.py) — training loop, evaluation, GIF export, plots
- [policy_eval.py](https://github.com/Jubzinas/Policy-Network/blob/main/policy_eval.py) — helper functions for evaluation and video export
- [setup.sh](https://github.com/Jubzinas/Policy-Network/blob/main/setup.sh) — creates the virtual environment and installs dependencies

To reproduce the run:

```bash
bash setup.sh
source .venv/bin/activate
python policy_network.py
```

---

## Summary

Policy networks are a direct way to learn behavior in reinforcement learning.
Instead of estimating action values first, the agent learns a distribution over
actions and improves that distribution using reward signals.

REINFORCE is one of the cleanest ways to understand that idea:
- sample actions from a policy
- compute returns
- increase the probability of good actions
- decrease the probability of bad ones

That makes it a natural next step after Q-Learning for understanding how modern
policy-gradient methods work.
