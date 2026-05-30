---
title: "Deep Q-Learning: Scaling RL to the Real World"
date: 2026-05-26
draft: false
tags: ["AI", "RL", "DQN", "Deep Learning"]
---

Tabular Q-Learning is elegant — but it breaks down fast. As soon as the
environment grows beyond a few hundred states, the Q-Table becomes impossibly
large to store and too slow to converge. A game like Breakout, played from
raw pixels, has an astronomically large state space: no table could ever hold it.

**Deep Q-Learning (DQN)** solves this by replacing the Q-Table with a
**neural network**. Instead of looking up a value in a table, the network
*approximates* the Q-function — taking a state as input and outputting
Q-values for every action in one forward pass.

$$Q_\theta(s, a) \approx Q^*(s, a)$$

The weights $\theta$ of the network are learned through gradient descent,
exactly like any other supervised learning problem — except the targets are
themselves computed from the network's own predictions.

---

## Why a Convolutional Neural Network?

For Atari games, the input to the agent is a **grayscale image** of the screen —
84×84 pixels, stacked across 4 consecutive frames to give the network a
sense of motion. This means the input tensor has shape **4 × 84 × 84**.

A plain fully-connected network would need to learn from every individual pixel
independently — no sense of spatial structure, no parameter sharing, millions
of weights. A **Convolutional Neural Network (CNN)** is a far better fit.

### What Is a Convolutional Layer?

A convolutional layer applies a set of small **filters** (also called kernels)
across the entire input image. Each filter slides over the image from left to
right, top to bottom, computing an **element-wise multiplication** between its
weights and the local patch of pixels underneath it. The sum of this operation
produces a single output pixel. Repeating this across the whole image produces
one **output channel** — also called a **feature map**.

$$\text{output}[i, j] = \sum_{m}\sum_{n} \text{input}[i \cdot s + m,\; j \cdot s + n] \cdot \text{kernel}[m, n]$$

where $s$ is the **stride** — how many pixels the filter shifts at each step.

<figure style="text-align:center; margin: 24px 0;">
  <img src="https://miro.medium.com/v2/resize:fit:1320/format:webp/1*LT0l-KXw5FXIkcGVl-KXlQ.gif"
       alt="Convolution kernel sliding over input" style="max-width:500px; width:100%; border-radius:8px;">
  <figcaption style="margin-top:8px; font-size:0.9em; color:#888;">A 3×3 kernel sliding over a 9×9 input (stride 1). Each position computes one output pixel via element-wise multiply + sum. <em>Source: <a href="https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148" target="_blank">Axel Thevenot</a></em></figcaption>
</figure>

A few key parameters control how a convolutional layer behaves
(as explained in depth by [Axel Thevenot](https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148)):

| Parameter | What it controls |
|-----------|-----------------|
| **Kernel size** | The spatial extent of the filter (e.g. 8×8, 4×4, 3×3). Larger kernels see more context but cost more compute. |
| **Stride** | How far the filter jumps between positions. A stride of 4 reduces output size by 4× — useful for downsampling. |
| **Number of filters** | Each filter learns a different feature (edges, curves, textures). More filters = more expressive, more parameters. |
| **Padding** | Zero-pixels added around the input border to preserve spatial size. |

<div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap; margin: 24px 0;">
  <figure style="text-align:center; flex:1; min-width:220px; max-width:340px;">
    <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ubRrYAZJUlCcqg7WoKjLgQ.gif"
         alt="Multi-channel convolution" style="width:100%; border-radius:8px;">
    <figcaption style="font-size:0.85em; color:#888; margin-top:6px;">Multiple filters on a 3-channel input → each output channel detects a different feature. <em>Source: <a href="https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148" target="_blank">Axel Thevenot</a></em></figcaption>
  </figure>
  <figure style="text-align:center; flex:1; min-width:220px; max-width:340px;">
    <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*o9-Rq3QUC8IzTMfAJIbhLA.gif"
         alt="Stride convolution" style="width:100%; border-radius:8px;">
    <figcaption style="font-size:0.85em; color:#888; margin-top:6px;">A large stride (e.g. 4 in Conv1) skips positions, aggressively downsampling the spatial size. <em>Source: <a href="https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148" target="_blank">Axel Thevenot</a></em></figcaption>
  </figure>
</div>

Each filter learns to detect a specific **visual pattern**. Early layers detect
edges and corners; deeper layers compose those into higher-level features
like shapes and objects. The network figures out which patterns are most useful
for predicting game outcomes entirely on its own.

### Output size formula

Given an input of height $H$, kernel size $K$, padding $P$, and stride $S$:

$$H_{\text{out}} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1$$

---

## The Nature DQN Architecture

The architecture used here is the original one from the DeepMind Nature paper
(Mnih et al., 2015). It processes the 4-frame stack through three convolutional
layers that progressively compress the spatial information into abstract features,
then feeds the result into two fully-connected layers that output one Q-value
per action.

| Layer | Specification | Output shape |
|-------|--------------|-------------|
| **Input** | 4 stacked grayscale frames ÷ 255 | 4 × 84 × 84 |
| **Conv1** | 32 filters, 8×8 kernel, stride 4, ReLU | 32 × 20 × 20 |
| **Conv2** | 64 filters, 4×4 kernel, stride 2, ReLU | 64 × 9 × 9 |
| **Conv3** | 64 filters, 3×3 kernel, stride 1, ReLU | 64 × 7 × 7 |
| **Flatten** | 64 × 7 × 7 = 3136 values | 3136 |
| **FC1** | 3136 → 512, ReLU | 512 |
| **FC2** | 512 → n_actions | n_actions |

After Conv3 the spatial maps are flattened into a single vector of **3136 values**
and fed into the fully-connected head. The final layer outputs one Q-value
per action — in Breakout, that's 4 values (no-op, fire, left, right).

---

## Key Innovations Over Tabular Q-Learning

Replacing the table with a network introduces instability — the network's own
outputs are used to compute the training targets. DQN solves this with two
critical tricks:

### 1. Experience Replay

Instead of learning from each transition immediately and discarding it, every
$(s, a, r, s')$ tuple is stored in a **replay buffer** (size 300,000 here).
At each training step, a random **mini-batch of 32** transitions is sampled
from the buffer.

This breaks the temporal correlation between consecutive samples — without it,
the network would overfit to whatever the agent is currently doing and quickly
diverge.

### 2. Target Network

A second copy of the Q-network — the **target network** — is kept frozen and
used to compute the TD targets:

$$y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

The target network's weights are synced to the online network every
**1,000 steps** (a hard copy, $\tau = 1.0$). This stabilises training by
preventing the targets from shifting at every gradient step.

---

## Training with Temporal Difference

The network is trained to minimise the **mean squared TD error** between its
predictions and the targets:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(y - Q_\theta(s, a)\right)^2\right]$$

$$y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

Gradients are backpropagated through the online network only — not through the
target. The optimizer used is **Adam** with learning rate $10^{-4}$.

---

## Exploration: ε-Greedy with Linear Decay

Just like tabular Q-Learning, DQN uses ε-greedy exploration — but the schedule
matters more here because the network needs many diverse experiences early on
to avoid collapsing to a suboptimal policy.

- **Start**: ε = 1.0 (fully random for the first 5,000 steps, before any training begins)
- **End**: ε = 0.01 (nearly fully greedy)
- **Decay**: linear over the first **10% of total timesteps**

---

## Atari Preprocessing

Raw Atari frames are 210×160 RGB images at 60fps — far too large and redundant
to train on directly. DeepMind's preprocessing pipeline reduces them to a compact,
informative representation:

1. **Grayscale** — color carries little information for game strategy.
2. **Resize to 84×84** — standardised input size across all games.
3. **Frame stacking (4 frames)** — gives the network temporal context:
   it can infer velocity of the ball, direction of movement, etc.
4. **Frame skip (4)** — the agent repeats each chosen action for 4 frames,
   reducing computational cost and making the control problem easier.
5. **Normalise to [0, 1]** — divide pixel values by 255.

<figure style="text-align:center; margin: 24px 0;">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/preprocessing.jpg"
       alt="Atari preprocessing pipeline: grayscale, resize, frame stacking"
       style="max-width:620px; width:100%; border-radius:8px;">
  <figcaption style="margin-top:8px; font-size:0.9em; color:#888;">The full DeepMind preprocessing pipeline: raw RGB frame → grayscale → 84×84 → stacked across 4 timesteps → 4×84×84 tensor fed to the CNN. — <em>Source: <a href="https://huggingface.co/learn/deep-rl-course/unit4/deep-q-network" target="_blank">Hugging Face Deep RL Course</a></em></figcaption>
</figure>

---

## Practical Implementation: Atari Breakout

I tested this implementation on `ALE/Breakout-v5` in [this repository](https://github.com/Jubzinas/Deep-Q-Learning),
training for **5 million steps** on a standard machine.

<figure style="text-align:center; margin: 24px 0;">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/deep-q-network.jpg"
       alt="Deep Q-Network architecture: CNN + fully connected layers"
       style="max-width:660px; width:100%; border-radius:8px;">
  <figcaption style="margin-top:8px; font-size:0.9em; color:#888;">The Nature DQN architecture: the stacked frames pass through 3 conv layers that extract spatial features, then 2 FC layers that output one Q-value per action. — <em>Source: <a href="https://huggingface.co/learn/deep-rl-course/unit4/deep-q-network" target="_blank">Hugging Face Deep RL Course</a>, based on Mnih et al. 2015</em></figcaption>
</figure>

The model implements the original Nature DQN with:
- CNN Q-network (3 conv layers + 2 FC layers)
- Experience replay buffer (300,000 transitions)
- Hard target network updates every 1,000 steps
- ε-greedy with linear decay
- Multi-environment rollouts (2 parallel envs)
- Periodic evaluation with video recording

### Before Training (random policy)

The agent takes random actions, scores close to 0, and loses lives immediately.

<div style="display:flex; justify-content:center; margin: 20px 0;">
  <figure style="text-align:center;">
    <img class="gif-loop" src="https://raw.githubusercontent.com/Jubzinas/Deep-Q-Learning/main/assets/before_training.gif" alt="Before training" style="max-width:480px; width:100%;">
    <figcaption>Before Training — random policy</figcaption>
  </figure>
</div>

### After 5M Steps

The agent has learned to aim the ball and systematically clear rows — showing
clearly purposeful, non-random behavior.

<div style="display:flex; justify-content:center; margin: 20px 0;">
  <figure style="text-align:center;">
    <img class="gif-loop" src="https://raw.githubusercontent.com/Jubzinas/Deep-Q-Learning/main/assets/after_training.gif" alt="After 5M steps" style="max-width:480px; width:100%;">
    <figcaption>After 5M Steps — learned policy</figcaption>
  </figure>
</div>

> Note: the original DeepMind paper trained for **50 million frames** to reach
> expert-level performance (scores of 400+). At 5M steps the agent is clearly
> learning but is still in an intermediate stage. To reproduce strong results,
> set `--total_timesteps 50000000` and use a GPU.

---

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Environment | `ALE/Breakout-v5` |
| Total timesteps | 5,000,000 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Parallel envs | 2 |
| Replay buffer size | 300,000 |
| Batch size | 32 |
| Discount factor (γ) | 0.99 |
| Target network update | every 1,000 steps (hard copy) |
| Training starts after | 5,000 steps |
| Train frequency | every 4 steps |
| ε start → end | 1.0 → 0.01 |
| Exploration fraction | 10% of timesteps |

> Full source code, setup instructions, and pretrained weights are available in
> the [GitHub repository](https://github.com/Jubzinas/Deep-Q-Learning).

---

## From Q-Learning to DQN: What Changed

| | Tabular Q-Learning | Deep Q-Learning |
|--|---|---|
| **State representation** | Discrete index | Raw pixels (tensor) |
| **Q-function** | Lookup table | Neural network |
| **Scalability** | Breaks at ~1000 states | Scales to millions |
| **Learning signal** | TD error per step | Batched MSE loss |
| **Stability tricks** | None needed | Replay buffer + target network |
| **Training** | Single update per step | Mini-batch gradient descent |

---

## Summary

Deep Q-Learning extends tabular Q-Learning to high-dimensional, visual environments by:

1. Replacing the Q-Table with a **CNN** that approximates $Q^*(s, a)$
2. Using an **experience replay buffer** to break temporal correlations
3. Stabilising training with a **frozen target network**
4. Processing raw pixels through **DeepMind's Atari preprocessing pipeline**

It was the first algorithm to demonstrate human-level performance across a broad
range of Atari games from raw pixel input — a landmark result published in
Nature in 2015 by Mnih et al.

<script>
(function() {
  function restartGifs() {
    document.querySelectorAll('img.gif-loop').forEach(function(img) {
      var src = img.src.split('?')[0];
      img.src = src + '?t=' + Date.now();
    });
  }
  setInterval(restartGifs, 8000);
})();
</script>
