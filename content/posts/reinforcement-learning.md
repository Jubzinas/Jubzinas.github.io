---
title: "Reinforcement Learning Introduction"
date: 2026-05-20
draft: false
tags: ["AI", "RL"]
---

Reinforcement Learning is a technique that allows an agent to learn from 
the environment by interacting with it, without any external instruction.
It is very similar to a kid trying to learn how to play a game without 
reading the instructions. He learns by actually playing, and every time 
he makes a mistake and the game restarts, he gets a little smarter.
An agent using Reinforcement Learning does the exact same thing.

<!--more-->

<img src="/images/rl_intro1.png" alt="Reinforcement Learning" style="width:100%; max-width:800px; display:block; margin: 20px auto;">

The difference is that the agent has a huge memory and can interact with 
multiple environments in parallel, learning from all of them at a much 
higher speed. As a result, agents trained via Reinforcement Learning can 
beat human players at games such as Chess and Go.

## How the System Works

The system is composed of five elements:
- An **agent**
- An **environment**
- A set of **actions** the agent can perform
- An **observation** the agent receives from the environment at each step
- A **reward** the agent receives from the environment at each step

At each step, the agent receives two things from the environment: a 
**state** (called an observation when the agent cannot see the full 
environment) and a **reward**. Based on the state, the agent decides 
which action to take. Each action leads to a new state and a specific 
reward.

At the beginning, the agent knows nothing about the environment, so it 
has no idea which reward or state to expect based on its actions.

The goal of the agent is to maximize the **expected return** — the sum 
of all future rewards, typically weighted by a discount factor so that 
immediate rewards are valued more than distant ones.