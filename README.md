# FLAX_RL: Yet another RL library build with FLAX modules

![Python Version](https://img.shields.io/badge/Python->=3.10-blue)
![Code Style](https://img.shields.io/badge/Code_Style-black-black)

[**Installation**](#installation) 
|  [**Overview**](#overview) 
| [**Example Usage**](#example-usage)
| [**Roadmap**](#example-usage)

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
import envpool
import ml_collections

from rl.algos import ppo
from rl.wrapper import EnvpoolCompatbility

N_ENVS = 32
N_ENV_STEPS = 10**5
CONFIG = ml_collections.ConfigDict(
    {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "_lambda": 0.95,
        "normalize": True,
        "max_buffer_size": 128,
        "batch_size": 256,
        "num_epochs": 1,
        "learning_rate_annealing": True,
        "max_grad_norm": 0.5,
        "n_env_steps": N_ENV_STEPS // N_ENVS,
        "shared_encoder": True,
        "save_frequency": -1,
    }
)
envs = EnvpoolCompatibility(
    envpool.make("CartPole-v1", env_type="gymnasium", num_envs=N_ENVS)
)
CONFIG["action_space"] = env.action_space
CONFIG["observation_space"] = env.observation_space

model = ppo.PPO(0, CONFIG, n_envs=N_ENVS, tabulate=True)
model.train(envs, CONFIG.n_env_steps, callbacks=[])
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