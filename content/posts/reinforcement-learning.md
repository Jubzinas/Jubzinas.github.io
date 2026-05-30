---
title: "Reinforcement Learning Introduction"
date: 2026-05-20
draft: false
tags: ["AI", "RL"]
---

Reinforcement Learning is a technique that allows an agent to learn from 
the environment by interacting with it, without any external instruction.
It is very similar to a kid trying to learn how to play a game without 
reading the instructions. The kid learns by actually playing, and every time 
he makes a mistake and the game restarts, he gets a little smarter.
An agent using Reinforcement Learning does the exact same thing — but faster.

The difference is that the agent has a huge memory and can interact with 
multiple environments in parallel, learning from all of them at a much 
higher speed. As a result, agents trained via Reinforcement Learning can 
beat human players at complex games such as Chess and Go.

<img src="/images/rl_intro1.png" alt="Reinforcement Learning diagram" style="width:100%; max-width:800px; display:block; margin: 20px auto;">

---

## How the System Works

At its core, a Reinforcement Learning system is a loop between an **agent** 
and an **environment**. At each step, the agent observes the environment, 
takes an action, and receives feedback in the form of a reward.

The system is composed of five key elements:

### 🤖 Agent
The learner and decision maker. It could be a neural network playing Atari, 
a robot learning to walk, or a program learning to trade stocks.

### 🌍 State and Observation
- A **state (S)** is a *complete* description of the environment — for example, 
the full board in a game of Chess where nothing is hidden.
- An **observation (o)** is a *partial* description — for example, in Super Mario 
the agent only sees the portion of the level currently on screen.

### 🎮 Action Space
The set of all possible actions the agent can perform. It can be:
- **Discrete** — a finite set of actions, such as: up, down, left, right.
- **Continuous** — actions drawn from a continuous range, such as steering 
angle and throttle in autonomous driving.

### 🏆 Reward
The only feedback signal the agent receives from the environment. 
A positive reward means the action was good, a negative reward means it was bad.
For example, in a game: +1 for scoring a point, -1 for losing a life, 0 otherwise.
The agent has no built-in knowledge — **the reward is everything**.

---

## The Training Loop

At each step the agent:
1. Observes the current state
2. Picks an action from its action space
3. Receives a new state and a reward from the environment
4. Updates its knowledge based on what it learned

This loop repeats until the **episode** ends.

---

## What is an Episode?

An **episode** is a complete sequence of interactions, from the initial 
state to a terminal state. Using our analogy — **one episode = one full game**.

It starts when the game begins and ends when one of two things happens:

- **Termination** — the agent reaches a natural end state. 
Example: the game is won or lost.
- **Truncation** — the episode is cut short after a maximum number of steps. 
Example: a time limit is reached even if the game isn't over.

Then the environment resets and a new episode begins. Over thousands of 
episodes, the agent gradually learns which actions lead to higher rewards.

---

## The Goal: Maximize Expected Return

At the beginning of training, the agent knows nothing — it has no idea 
which reward or state to expect based on its actions. It explores randomly 
at first, then gradually learns to prefer actions that lead to higher rewards.

The goal is to maximize the **expected return** — the sum of all future rewards:

$$G_t = r_{t+1} + r_{t+2} + r_{t+3} + \dots$$

In practice, a **discount factor γ (gamma)** between 0 and 1 is applied, 
so that immediate rewards are valued more than distant ones:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$$

A γ close to 1 means the agent is *far-sighted* (cares about future rewards).
A γ close to 0 means the agent is *short-sighted* (only cares about immediate reward).