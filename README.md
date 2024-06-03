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

> [!IMPORTANT]
> New reinforcement learning algorithms are frequently added to this project. However, for benchmarking purposes, please refer to the original implementations.

> [!NOTE]
> The following README is an overview of what the library offers. Please refer to the [documentation](https://raffaelbdl.github.io/kitae/) for more details.

Kitae aims to be a middle ground between 'clear RL' implementations and 'use-only' libraries (SB3, ...).

In Kitae, an Agent is entirely defined by a configuration and 4 factory functions: 
- `train_state_factory`: creates the Agent's state
- `explore_factory`: creates the function used to interact in the environment
- `process_experience_factory`: creates the function to process the data before updating
- `update_step_factory`: creates the function to update the Agent's state

These functions can be implemented very closely to 'clean RL' implementations, but are ultimately encapsulated into a single class which simplifies the use of multiple environments, saving and loading, etc...

Kitae offers a few tools to simplify writing agents. In particular, self-play in multi-agent settings and vectorized environments are automatically handled by the library.

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
from kitae.algos.collections import ppo
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
        seed=SEED,
        algo_params=ppo.PPOParams(
            gamma=0.99,
            _lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            normalize=True,
        ),
        update_cfg=cfg.UpdateConfig(
            learning_rate=0.0003,
            learning_rate_annealing=True,
            max_grad_norm=0.5,
            max_buffer_size=64,
            batch_size=256,
            n_epochs=1,
            shared_encoder=True,
        ),
        train_cfg=cfg.TrainConfig(n_env_steps=5*10**5, save_frequency=-1),
        env_cfg=env_config,
    ),
    tabulate=True,
)

algo.train(env, algo.config.train_cfg.n_env_steps)
```

## How to write a custom agent with Kitae

The process of building a custom agent is detailed in this [Google Colab](https://colab.research.google.com/drive/1pm542fVcqnct5LDfeovM9mxDqWjAMscr?usp=sharing).