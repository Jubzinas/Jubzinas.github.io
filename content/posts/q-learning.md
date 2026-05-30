---
title: "Q-Learning: Teaching Machines to Make Decisions"
date: 2026-05-26
draft: false
tags: ["AI", "RL", "Q-Learning"]
---

Q-Learning is one of the most fundamental algorithms in Reinforcement Learning.
It allows an agent to learn the optimal strategy for navigating an environment
purely through trial and error — no model of the world required.

Think of it like a traveler exploring a city without a map. Every time they find
a shortcut or hit a dead end, they update their mental notes. Over time, those
notes become a reliable guide to reaching any destination as efficiently as possible.

It belongs to a specific family of RL methods — and understanding *where* it sits
in that family helps understand why it works the way it does.

<figure style="text-align:center; margin: 28px 0;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/Reinforcement_learning_diagram.svg"
       alt="Reinforcement Learning agent-environment loop"
       style="max-width:520px; width:100%; background:#fff; padding:12px; border-radius:8px;">
  <figcaption style="margin-top:8px; font-size:0.9em; color:#888;">The RL loop: the agent observes a state, takes an action, and receives a reward from the environment.</figcaption>
</figure>

---

## Q-Learning is a Value-Based Method

Reinforcement Learning algorithms can be broadly split into two families:

- **Policy-based** methods directly learn a function that maps states to actions.
- **Value-based** methods instead learn to estimate *how good* each situation is,
  and derive the policy from those estimates.

Q-Learning is **value-based**. It never explicitly stores a policy. Instead, it
learns a **value function** — and the policy is implicit: always pick the action
with the highest value.

---

## The Q-Table: Shape and Meaning

At the heart of Q-Learning is the **Q-Table**, a 2D matrix with a precise structure:

$$Q \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$$

- **Rows** → every possible **state** $s \in \mathcal{S}$
- **Columns** → every possible **action** $a \in \mathcal{A}$
- **Each cell** $Q(s, a)$ → the estimated **total future reward** the agent
  expects to collect, starting from state $s$, taking action $a$, and then
  following the best possible policy from that point onward.

For example, in the **Taxi** environment there are 500 possible states and
6 possible actions, so the Q-Table has shape **500 × 6 = 3,000 values** to learn.
In **FrozenLake** (8×8 grid), it's **64 × 4 = 256 values**.

Initially every cell is set to zero — the agent knows nothing. The entire
learning process is about filling these cells with increasingly accurate estimates.

To make it concrete, here is what a small converged Q-Table looks like for a 4-state GridWorld with 4 actions:

| State | ← Left | ↓ Down | → Right | ↑ Up |
|-------|--------|--------|---------|------|
| S (start) | 0.31 | 0.48 | **0.72** | 0.19 |
| State 1 | −0.10 | **0.65** | 0.41 | −0.05 |
| State 2 | −0.20 | 0.12 | **0.80** | 0.22 |
| G (goal) | 0.00 | 0.00 | 0.00 | 0.00 |

The **bold** value in each row is the action the agent will take — that is the implicit policy.

---

## What Exactly Is a Q-Value?

The Q-value $Q(s, a)$ is not just the next immediate reward. It is the
**expected discounted return** — the sum of all future rewards the agent will
collect from this moment onward, with rewards further in the future discounted
by a factor $\gamma$:

$$Q(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \, r_{t+1} \;\bigg|\; s_0 = s,\, a_0 = a \right]$$

The discount factor $\gamma \in [0, 1]$ controls how far-sighted the agent is:
- $\gamma = 0$: the agent only cares about the immediate reward.
- $\gamma \to 1$: the agent treats near and distant rewards almost equally.
- A typical value like $\gamma = 0.99$ means a reward 100 steps away is still
  worth about $0.99^{100} \approx 37\%$ of a reward received now.

This is what separates Q-Learning from a simple reflex agent — it plans ahead.

---

## How Q-Learning Learns: Temporal Difference

Q-Learning belongs to the **Temporal Difference (TD)** family of learning methods.
The key idea of TD learning is: **don't wait until the end of an episode to update**.
Instead, update after *every single step*, using the very next observation as a
bootstrapped estimate of the future.

After taking action $a$ from state $s$, the agent:
1. Receives immediate reward $r$
2. Lands in next state $s'$
3. Looks up the best Q-value it currently knows for $s'$: $\max_{a'} Q(s', a')$
4. Constructs a **TD target** — a better estimate of what $Q(s, a)$ should be:

$$\text{TD target} = r + \gamma \max_{a'} Q(s', a')$$

This target combines the *real* reward just received with the *current best guess*
of future value. The difference between this target and the current estimate is
called the **TD error**:

$$\delta = \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD target}} - \underbrace{Q(s, a)}_{\text{current estimate}}$$

A large $\delta$ means the agent was surprised — its current estimate was far off.
A small $\delta$ means it already had a good estimate of that state-action pair.

<figure style="text-align:center; margin: 24px 0;">
  <img src="https://gymnasium.farama.org/_images/frozen_lake.gif"
       alt="FrozenLake Q-Learning agent navigating the grid"
       style="max-width:360px; width:100%; border-radius:8px; image-rendering:pixelated;">
  <figcaption style="margin-top:8px; font-size:0.9em; color:#888;">A trained Q-Learning agent navigating FrozenLake. After thousands of TD updates, the Q-Table has converged enough to guide the agent safely from S to G. — <em>Source: <a href="https://gymnasium.farama.org/environments/toy_text/frozen_lake/" target="_blank">Gymnasium — FrozenLake-v1</a></em></figcaption>
</figure>

---

## The Bellman Equation and the Update Rule

The update rule of Q-Learning is rooted in the **Bellman Optimality Equation**,
which expresses a fundamental recursive truth: *the value of a state is equal to
the best immediate reward you can get, plus the discounted value of the best
next state you can reach.*

$$Q^*(s, a) = r(s, a) + \gamma \max_{a'} Q^*(s', a')$$

This is not just a formula — it's a *consistency condition*. If the Q-values are
truly optimal ($Q^*$), this equation must hold for every state-action pair.
Q-Learning exploits this: it repeatedly nudges the current estimates toward
satisfying it, using the update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \delta$$

Expanded in full:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Let's break down every component:

| Symbol | Name | Role |
|--------|------|------|
| $Q(s, a)$ | Current Q-value | Our existing estimate for this state-action pair |
| $\alpha \in (0,1]$ | Learning rate | Step size of the update — how aggressively we revise the estimate |
| $r$ | Immediate reward | The real signal received from the environment after action $a$ |
| $\gamma \in [0,1)$ | Discount factor | How much future rewards are worth relative to immediate ones |
| $s'$ | Next state | The state the environment transitioned to after the action |
| $\max_{a'} Q(s', a')$ | Best future value | The highest Q-value in the next state — our bootstrapped future estimate |
| $\delta$ | TD error | The gap between what we expected and what we now believe is correct |

The learning rate $\alpha$ controls how large a correction we make:
- $\alpha = 1$: fully replace the old estimate with the new target (aggressive).
- $\alpha \to 0$: barely update at all (very conservative).
- In practice values like $0.1$ work well — we nudge the estimate a little each step.

Over thousands of steps, these small corrections accumulate until $Q(s,a)$
converges to $Q^*(s,a)$ — the true optimal value — for every state-action pair.

---

## Exploration vs. Exploitation

A key challenge in Q-Learning is deciding when to **explore** (try new actions)
and when to **exploit** (use what's already learned). This is managed through
the **ε-greedy policy**:

- With probability **ε**: pick a **random action** (explore)
- With probability **1 - ε**: pick the **best known action** (exploit)

Early in training, ε is kept high so the agent explores broadly.
Over time, ε is gradually reduced — the agent shifts toward exploiting
the knowledge it has accumulated.

| Phase | ε value | Behavior |
|-------|----------|----------|
| Start of training | ≈ 1.0 | Almost always picks a **random action** — full exploration |
| Mid training | ≈ 0.5 | Mix of random and greedy choices |
| End of training | ≈ 0.0001 | Almost always picks the **best known action** — full exploitation |

ε decays linearly over training. This schedule ensures the agent sees diverse situations early on before committing to the policy it has learned.

---

## A Simple Example: GridWorld

Imagine a 4×4 grid. The agent starts at the top-left corner and must reach
a goal in the bottom-right corner, avoiding a pit in the middle.

```
S . . .
. . X .
. . . .
. . . G
```

- `S` = Start, `G` = Goal, `X` = Pit (negative reward)

At each step, the agent can move **up, down, left, or right**.
It receives:
- `+10` for reaching the goal
- `-10` for falling into the pit
- `-1` for every other step (to encourage efficiency)

After thousands of episodes, the Q-Table will have converged to values that
guide the agent along the optimal path from any cell on the grid.

---

## Limitations of Q-Learning

Q-Learning works well for small, discrete environments — but it has real limits:

- **Scalability**: The Q-Table grows with every new state and action.
  For complex environments (like video games), it becomes unmanageable.
- **Continuous spaces**: Q-Learning can't handle continuous state or action spaces
  without discretization, which introduces approximation errors.

These limitations led to the development of **Deep Q-Networks (DQN)**, where
a neural network replaces the Q-Table — enabling Q-Learning to scale to
environments like Atari games with raw pixel input.

---

## Practical Implementation: Two Environments

To see Q-Learning in action, I tested it on two classic [Gymnasium](https://gymnasium.farama.org/)
environments in [this repository](https://github.com/Jubzinas/Q-learning).
Both agents learn purely from scratch using epsilon-greedy exploration
and are periodically evaluated during training.

---

### 🧊 FrozenLake

The agent navigates an **8×8 frozen grid** from the start tile (S) to the goal (G),
avoiding holes (H). A random map is generated each run, and the environment
supports both slippery and non-slippery modes.

- **State space**: 64 discrete positions
- **Action space**: 4 (Left, Down, Right, Up)
- **Reward**: +1 for reaching the goal, 0 otherwise
- **Metric tracked**: Success rate (% of episodes reaching the goal)

<div style="display:flex; gap:16px; justify-content:center; text-align:center; margin: 20px 0;">
  <figure>
    <img class="gif-loop" src="https://raw.githubusercontent.com/Jubzinas/Q-learning/main/assets/frozen_lake_pretrained.gif" alt="FrozenLake untrained" style="max-width:300px; width:100%;">
    <figcaption>Before Training</figcaption>
  </figure>
  <figure>
    <img class="gif-loop" src="https://raw.githubusercontent.com/Jubzinas/Q-learning/main/assets/frozen_lake_trained.gif" alt="FrozenLake trained" style="max-width:300px; width:100%;">
    <figcaption>After Training</figcaption>
  </figure>
</div>

The untrained agent stumbles randomly and almost always falls into a hole.
After training, it reliably finds a safe path to the goal.

---

### 🚕 Taxi

The agent drives a taxi on a **5×5 grid**, picking up a passenger and dropping
them off at the correct destination.

- **State space**: 500 discrete states (taxi position × passenger location × destination)
- **Action space**: 6 (South, North, East, West, Pickup, Dropoff)
- **Reward**: +20 for a correct dropoff, −1 per step, −10 for illegal actions
- **Metric tracked**: Mean reward per evaluation episode

<div style="display:flex; gap:16px; justify-content:center; text-align:center; margin: 20px 0;">
  <figure>
    <img class="gif-loop" src="https://raw.githubusercontent.com/Jubzinas/Q-learning/main/assets/taxi_pretrained.gif" alt="Taxi untrained" style="max-width:300px; width:100%;">
    <figcaption>Before Training</figcaption>
  </figure>
  <figure>
    <img class="gif-loop" src="https://raw.githubusercontent.com/Jubzinas/Q-learning/main/assets/taxi_trained.gif" alt="Taxi trained" style="max-width:300px; width:100%;">
    <figcaption>After Training</figcaption>
  </figure>
</div>

Before training the agent makes random illegal moves and ignores the passenger.
After training it navigates directly to the pickup, then to the correct dropoff location.

---

### Hyperparameters used

Both agents were trained with the following configuration:

| Hyperparameter | Value |
|---|---|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.99 |
| Min epsilon | 1e-4 |
| Max steps per episode | 200 |

Epsilon decays linearly from 1.0 to `min_epsilon` over all training episodes.

> The full source code, setup instructions, and auto-generated GIFs are available
> in the [GitHub repository](https://github.com/Jubzinas/Q-learning).

---

## Summary

Q-Learning is an elegant algorithm that:

1. Maintains a table of state-action values
2. Updates them using the Bellman equation after each step
3. Balances exploration and exploitation with ε-greedy
4. Converges to the optimal policy given enough time and exploration

It laid the groundwork for modern deep reinforcement learning and remains
one of the most important ideas in the field.

<script>
(function() {
  function restartGifs() {
    document.querySelectorAll('img.gif-loop').forEach(function(img) {
      var src = img.src.split('?')[0];
      img.src = src + '?t=' + Date.now();
    });
  }
  // Restart every 6 seconds — adjust to match your GIF duration
  setInterval(restartGifs, 6000);
})();
</script>
