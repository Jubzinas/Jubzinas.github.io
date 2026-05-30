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

---

## Why Use a Policy Network?

Policy networks are useful when:

- you want to learn the action choice directly
- you need stochastic policies for exploration
- the environment is continuous or high-dimensional
- you care about optimizing long-term return rather than local action values

Instead of building a Q-table, the network learns parameters that map states
straight to action probabilities.

---

## REINFORCE: The Core Idea

REINFORCE is a Monte Carlo policy gradient method. It waits until an episode
(or rollout) is finished, computes the return, and then updates the policy so
that actions that led to good outcomes become more likely.

The objective is to maximize expected return:

$$J(\theta) = \mathbb{E}[G]$$

where $G$ is the total discounted reward.

The policy gradient theorem gives the update direction:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t \right]$$

In practice, this means:

- if an action leads to a good return, increase its probability
- if an action leads to a bad return, decrease its probability

A common training loss is the negative of that objective:

$$\mathcal{L}(\theta) = -\log \pi_\theta(a_t \mid s_t) \cdot G_t$$

This is why policy gradients are often described as learning by
“reinforcing actions that worked.”

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
