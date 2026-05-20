---
title: "Reinforcement Learning"
date: 2026-05-20
draft: false
tags: ["AI", "RL"]
---
Reinforcement Learning is a techique that permits an agent to learn from the environment by interacting with it, with no external instruction.
It is like a kid that is trying to learn how to play a game, without reading instructions by trial and error. He understand how to play and learn every time it restarts the game.
An agent that is using reinforcement learning is doing the exact same thing.
The difference is that the agent has a huge memory and can interact with multiple environments in parallel (and learning from all of them) with a speed way more high.
The result is that agents trained via reinforcement learning can beat players at games such as chess and go. 

The system is set in the same way:
We have an agent, an environment, a set of actions that the agent can do, an observation that the agent receive from the environment at each step and a reward that the agent receive from the environment at each step.

So the agent gets 2 elements from the environment: a state (if it is not the ful environment it is called observation) and a reward.
Based on the state the agent can decide which action to do from the set of actions that he owns.
Each action lead to a new state and a specific reward.
At the beginning the agents does not know anything about the environment so do not know which reward and state to expect, base on its action.

The goal of the agent is to maximize the expected return, so the sum of all the expected rewards.
