<div align="center">
  <img src="./docs/source/_static/images/sword.png" />
</div>

# Kitae: Yet another RL library built with FLAX modules

![Tests](https://github.com/Raffaelbdl/rl_tools/actions/workflows/pytest.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-Work_In_Progress-orange)

![Python Version](https://img.shields.io/badge/Python->=3.10-blue)
![Code Style](https://img.shields.io/badge/Code_Style-black-black)

[**Installation**](#installation) 
|  [**Overview**](#overview) 
| [**Example Usage**](#example-usage)
<!-- | [**Roadmap**](#roadmap) -->

Kitae aims to be a general toolbox to make working with reinforcement learning easier.

> [!IMPORTANT]
> New reinforcement learning algorithms are frequently added to this project. However, for benchmarking purposes, please refer to the original implementations.

> [!NOTE]
> The following README is an overview of what the library offers. Please refer to the [documentation](https://raffaelbdl.github.io/kitae/) for more details.



## Installation
This package requires Python 3.10 or later and a working [JAX](https://github.com/google/jax) installation.
To install JAX, refer to [the instructions](https://github.com/google/jax#installation).

```bash
pip install --upgrade pip
pip install --upgrade git+https://github.com/Raffaelbdl/kitae
```

## Overview

Kitae is designed as a polyvalent toolbox library for reinforcement learning. The goal is to simplify all steps of the process, from agent creation, to training and evaluating them.

One main feature of Kitae, is that it is designed to simplify working in vectorized settings with multiple instances of a environment.

## Example Usage
```python
from kitae.algos.named import ppo
from kitae import config as cfg
from kitae.envs.make import make_vec_env

SEED = 0
ENV_ID = "CartPole-v1"

env = make_vec_env(ENV_ID, 16, capture_video=False, run_name=None)
env_cfg = cfg.EnvConfig(
    ENV_ID, 
    env.single_observation_space, 
    env.single_action_space, 
    n_envs=16, 
    n_agents=1
)

agent = ppo.PPO(
    "example-ppo",
    cfg.AlgoConfig(
        SEED,
        ppo.PPOParams(
            gamma=0.99,
            _lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            normalize=True,
        ),
        cfg.UpdateConfig(
            learning_rate=0.0003,
            learning_rate_annealing=True,
            max_grad_norm=0.5,
            max_buffer_size=64,
            batch_size=256,
            n_epochs=1,
            shared_encoder=True,
        ),
        cfg.TrainConfig(n_env_steps=5*10**5, save_frequency=-1),
        env_config,
    ),
    tabulate=True,
)

algo.train(env, algo.config.train_cfg.n_env_steps, callbacks=[])
```

<!-- ## Roadmap

This project is still a work in progress. The current version is v0.1.0:
- [x] Implementation of state-of-the-art algorithms:
    - [x] PPO 
    - [x] SAC
    - [x] TD3
- [x] Automatic adaptation to multi-agent (self-play) and vectorial environments. 
- [x] Training loops, callbacks etc..
 -->
