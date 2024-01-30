# FLAX_RL: Yet another RL library built with FLAX modules

![Python Version](https://img.shields.io/badge/Python->=3.10-blue)
![Code Style](https://img.shields.io/badge/Code_Style-black-black)

[**Installation**](#installation) 
|  [**Overview**](#overview) 
| [**Example Usage**](#example-usage)
| [**Roadmap**](#roadmap)

## Installation
This package requires Python 3.10 or later and a working [JAX](https://github.com/google/jax) installation.
To install JAX, refer to [the instructions](https://github.com/google/jax#installation).

```bash
pip install --upgrade pip
pip install --upgrade git+https://github.com/Raffaelbdl/flax_rl
```

## Overview

FLAX_RL is designed as a polyvalent RL library. The goal is to make agents creation easier, as well as training and evaluating. 

FLAX_RL's goals is also to make it easier to use agents in single or multi-agent setups, as well as vectorized environments.

For research purposes, please use the original implementations for comparison. 


## Example Usage
```python
from evals.eval_envs import make_pong_vector

from rl.algos import ppo
from rl import config as cfg

SEED = 0

env, env_config = make_pong_vector(32)

algo = ppo.PPO(
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
            max_buffer_size=128,
            batch_size=256,
            n_epochs=1,
            shared_encoder=True,
        ),
        cfg.TrainConfig(n_env_steps=5*10**5, save_frequency=-1),
        env_config,
    )
)

algo.train(env, algo.config.train_cfg.n_env_steps, callbacks=[])
```

## Roadmap

This project is still a work in progress. The current version is v0.1.0:
- [x] Implementation of state-of-the-art algorithms:
    - [x] PPO 
    - [x] SAC
    - [x] TD3
- [x] Automatic adaptation to multi-agent (self-play) and vectorial environments. 
- [x] Training loops, callbacks etc..

