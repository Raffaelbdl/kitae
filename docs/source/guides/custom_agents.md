# Custom Agents

Agents in **Kitae** follows the `IAgent` interface. Most of the functionalities are already pre-implemented by the `BaseAgent` class, from which we will derive a DQN Agent in this tutorial.

To simplify the implementation of agents, **Kitae** uses 4 factory functions:
- `train_state_factory`: creates the agent's states
- `explore_factory`: creates the function used to interact in the environment
- `process_experience_factory`: creates the function to process the data before updating
- `update_step_factory`: creates the function to update the agent's states

## 0. Import everything
```python
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import distrax as dx
import jax
import jax.numpy as jnp
import optax

from kitae.base import OffPolicyAgent
import kitae.config as cfg
from kitae.types import Params

from kitae.buffer import Experience
from kitae.loss import loss_mean_squared_error
from kitae.timesteps import compute_td_targets

from kitae.modules.modules import init_params
from kitae.modules.train_state import TrainState
from kitae.modules.qvalue import qvalue_factory

DQN_tuple = namedtuple("DQN_tuple", ["observation", "action", "return_"])
NO_EXPLORATION = 0.0
```

## 1. DQN parameters 

```python
@dataclass
class DQNParams:
    exploration: float
    gamma: float
    skip_steps: int
    start_step: int = -1
```

An `AlgoParams` is a simple instance of a dataclass. Inheriting from `kitae.config.AlgoParams` is optional.

## 2. TrainState Factory

The `train_state_factory` takes a `key` and an `AlgoConfig` as arguments. 

Its output will be stored in the agent as `state` attribute.

```python
def train_state_dqn_factory(
    key: jax.Array,
    config: cfg.AlgoConfig,
    *,
    preprocess_fn: Callable = None,
    tabulate: bool = False,
) -> TrainState:
    observation_shape = config.env_cfg.observation_space.shape
    n_actions = config.env_cfg.action_space.n   # discrete spaces only

    class QValue(nn.Module):
        @nn.compact
        def __call__(self, observations: jax.Array) -> jax.Array:
            x = observations
            x = nn.relu(nn.Dense(64)(x))
            x = nn.relu(nn.Dense(64)(x))
            return nn.Dense(n_actions)(x)
    
    qvalue = QValue()
    return TrainState.create(
        apply_fn=qvalue.apply,
        params=init_params(key, qvalue, [observation_shape], tabulate),
        target_params=init_params(key, qvalue, [observation_shape], False),
        tx=optax.adam(config.update_cfg.learning_rate),
    )
```

Here we use `kitae.modules.train_state.TrainState` which has an additional `target_params` attribute. In this case, a `flax.training.train_state.TrainState` would have been enough.

## 3. Explore factory

The `explore_factory` takes an `AlgoConfig` as argument. 

Its output is a function that takes a state, a `key` and a number of trees as positional arguments. This function should return two Array, an `action` and a `log_prob` associated to the action.

This function should consider inputs of the shape `[batch_size, ...]`.

```python
def explore_factory(config: cfg.AlgoConfig) -> Callable:
    @jax.jit
    def explore_fn(
        dqn_state: TrainState,
        key: jax.Array,
        observations: jax.Array,
        exploration: float,
    ) -> jax.Array:
        all_qvalues = dqn_state.apply_fn(dqn_state.params, observations)
        actions, log_probs = dx.EpsilonGreedy(
            all_qvalues, exploration
        ).sample_and_log_prob(seed=key)

        return actions, log_probs

    return explore_fn
```

## 4. Process experience factory

The `process_experience_factory` takes an `AlgoConfig` as argument. 

Its output is a function that takes a state, a `key` and a tuple of Arrays. This function should return a tuple of Arrays after processing the inputs.

This function should consider inputs of the shape `[batch_size, ...]`.

```python
import jax.numpy as jnp

from kitae.buffer import Experience

def process_experience_factory(config: cfg.AlgoConfig) -> Callable:
    algo_params = config.algo_params

    @jax.jit
    def process_experience_fn(
        dqn_state: TrainState,
        key: jax.Array,
        experience: Experience,
    ) -> tuple[jax.Array, ...]:

        all_next_qvalues = dqn_state.apply_fn(
            dqn_state.params, experience.next_observation
        )
        next_qvalues = jnp.max(all_next_qvalues, axis=-1, keepdims=True)

        discounts = algo_params.gamma * (1.0 - experience.done[..., None])
        returns = rewards[..., None] + discounts * next_values
        
        actions = experience.action[..., None]

        return (experience.observation, actions, returns)

    return process_experience_fn
```

## 5. Update factory

The `update_step_factory` takes an `AlgoConfig` as argument. 

Its output is a function that takes a state, a `key` and a tuple of Arrays. This function should return a updated version of the state, and a dictionary with the update information.

This function should consider inputs of the shape `[batch_size, ...]`.

```python
def update_step_factory(config: cfg.AlgoConfig) -> Callable:

    @jax.jit
    def update_step_fn(
        dqn_state: TrainState,
        key: jax.Array,
        experiences: tuple[jax.Array, ...],
    ) -> tuple[TrainState, dict]:
        batch = DQN_tuple(*experiences)

        def loss_fn(params: Params):
            all_qvalues = qvalue_state.apply_fn(params, batch.observation)
            qvalues = jnp.take_along_axis(all_qvalues, batch.action, axis=-1)
            loss = jnp.mean(jnp.sum(jnp.square(qvalues - batch.return_), axis=-1))
            return loss, {"loss_qvalue": loss}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            qvalue_state.params
        )
        qvalue_state = qvalue_state.apply_gradients(grads=grads)
        return qvalue_state, info

    return update_step_fn
```

## 6. The DQN class

Thanks to the `kitae.base.OffPolicyAgent` class, only the `explore` and `select_action` methods need to be implemented. 

```python
class DQN(OffPolicyAgent):
    def __init__(
        self,
        run_name: str,
        config: cfg.AlgoConfig,
        *,
        preprocess_fn: Callable = None,
        tabulate: bool = False,
    ):
        super().__init__(
            run_name,
            config,
            train_state_dqn_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=Experience,
        )

        self.algo_params = self.config.algo_params

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)

        action, zeros = self.explore_fn(
            self.state, keys, observation, exploration=NO_EXPLORATION
        )
        return action, zeros

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)

        action, zeros = self.explore_fn(
            self.state,
            keys,
            observation,
            exploration=self.algo_params.exploration,
        )
        return action, zeros
```

## 7. Training

You can now instantiate and train your DQN!

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

env_cfg = cfg.EnvConfig(
    task_name=ENV_ID,
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    n_envs=N_ENVS,
    n_agents=1
)

dqn_params = DQNParams(
    exploration=0.1,
    gamma=0.99,
    skip_steps=1,
    start_step=-1
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
    algo_params=dqn_params,
    update_cfg=update_cfg,
    train_cfg=train_cfg,
    env_cfg=env_cfg,
)

dqn = DQN(run_name="example-dqn", config=algo_config)
dqn.train(env, dqn.config.train_cfg.n_env_steps)
```
