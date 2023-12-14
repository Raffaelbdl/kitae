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
FLAX_RL is a simple RL library with the goal of making it easy to use agents after training. To achieve that goal, all agents observe the `Base` abstract class implementation.

FLAX_RL also aims at making it easier to use agents in single or multi-agents setups. It also aims at simplifying parallelized training. 

For research purposes, please use the original implementations for comparaison. 

<details>
<summary>Implemented algorithms</summary>
- PPO
- IPPO
- DQN
</details>



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
Current roadmap to reach V0.1.0 :
- [ ] : Implement other algorithms
    - [ ] : PPO in continous action space
    - [ ] : SAC
    - [ ] : DDPG
- [ ] : Automate the multi-agents versions of every algorithms
- [ ] : Move the populations to another library
- [ ] : Benchmark the performance of the algorithms
- [ ] : Make a reliable config system with comprehensive saving and loading for simpler loading of individual agents when not training
- [ ] : Add docstrings
