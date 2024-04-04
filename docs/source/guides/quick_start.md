# Quick Start

This guide will show you how to use existing agents. To implement custom agents, please see [Custom Agents](custom_agents.md).

## 1. Installation

This package requires Python 3.10 or later and a working [JAX](https://github.com/google/jax) installation.
To install JAX, refer to [the instructions](https://github.com/google/jax#installation).

```bash
pip install --upgrade pip
pip install --upgrade git+https://github.com/Raffaelbdl/flax_rl
```

## 2. Create an environment

**Kitae** training loop uses vectorized environments. You can simply load one using **Kitae**'s `make` functions.

```python
from kitae.envs.make import make_vec_env

SEED = 0
ENV_ID = "CartPole-v1"
N_ENVS = 16

env = make_vec_env(
    env_id=ENV_ID, 
    n_envs=N_ENVS, 
    capture_video=False, 
    run_name=None,
)
```

Here we create a CartPole environment, vectorized on 16 processes.
If you only need 1 environment, you can either set `n_envs=1`, or use the `make_single_env` function.

## 3. Create an AlgoConfig

All parameters are encapsulated into an `AlgoConfig`. 

An `AlgoConfig` is composed of 4 subconfigs:
- `AlgoParams`: the algorithm specific parameters
- `UpdateConfig`: the update specific parameters
- `TrainConfig`: the training specific parameters
- `EnvConfig`: the environment specific parameters
  
In this guide, we will use a ppo instance.

```python
from kitae import config as cfg
from kitae.algos import ppo

env_cfg = cfg.EnvConfig(
    task_name=ENV_ID,
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    n_envs=N_ENVS,
    n_agents=1
)

algo_params = ppo.PPOParams(
    gamma=0.99,
    _lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    normalize=True,
)

update_cfg = cfg.UpdateConfig(
    learning_rate=0.0003,
    learning_rate_annealing=True,
    max_grad_norm=0.5,
    max_buffer_size=64,
    batch_size=256,
    n_epochs=1,
    shared_encoder=True,
)

train_cfg = cfg.TrainConfig(n_env_steps=5*10**5, save_frequency=-1)

algo_config = cfg.AlgoConfig(
    seed=SEED,
    algo_params=algo_params,
    update_cfg=update_cfg,
    train_cfg=train_cfg,
    env_cfg=env_cfg,
)
```

## 4. Instantiate and train an agent

Finally, let's instantiate a PPO agent.

```python
agent = ppo.PPO(run_name="example-ppo", config=algo_config)
```

Once a config has been defined, instantiating an agent is as simple as that! Training it is no more complex.

```python
agent.train(env, agent.config.train_cfg.n_env_steps)
```

