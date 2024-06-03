from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import chex
import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray

from jrd_extensions import PRNGSequence

from kitae.pytree import AgentPyTree


ExperienceTransform = Callable[[AgentPyTree, PRNGKeyArray, NamedTuple], NamedTuple]


# When jitting, n should be specified as a static argument
def merge_n_first_dims(array: jax.Array, n: int = 2) -> jax.Array:
    """Merges the n-first dimensions of an Array.

    Eg:
        n==1: (T, E, ...) => (T, E, ...)
        n==2: (T, E, ...) => (B * E, ...)
        n==3: (T, E, A, ...) => (B * E * A, ...)

    Raises:
        AssertionError: If n is lower or equal to 0.
        AssertionError: If n is not specified as static when jitting.
    """
    chex.assert_scalar_non_negative(n - 1)
    return jnp.reshape(array, (-1, *array.shape[n:]))


# When jitting, n should be specified as a static argument
def stack_and_merge_n_first_dims(arrays: Sequence[jax.Array], n: int = 2) -> jax.Array:
    """Stacks a sequence of Arrays then merges its n-first dimensions.

    Eg:
        n==1: A * (T, E, ...) => (T, A, E, ...)
        n==2: A * (T, E, ...) => (B * A, E, ...)
        n==3: A * (T, E, ...) => (B * A * E, ...)

    Raises:
        AssertionError: If n is lower or equal to 0.
        AssertionError: If n is not specified as static when jitting.
    """
    return merge_n_first_dims(jnp.stack(arrays, axis=1), n)


def tuple_to_dict(experience: NamedTuple) -> dict[str, NamedTuple]:
    """Converts a NamedTuple of dictionaries to a dictionary of NamedTuple.

    Eg:
        `Foo(a={"a": Array_a, "b": Array_b})` becomes:
        `{"a": Foo(a=Array_a), "b": Foo(b=Array_b)}`
    """

    _cls = experience.__class__

    output = {}
    for k in experience[0].keys():
        output[k] = _cls(*[e[k] for e in experience])

    return output


def dict_to_tuple(experience: dict[str, NamedTuple]) -> NamedTuple:
    # {"a": Experience(a=Array_a), "b": Experience(b=Array_b)}
    # => Experience(a={"a": Array_a, "b": Array_b})
    first_named_tuple = next(iter(experience.values()))
    _cls = first_named_tuple.__class__

    _args = [{} for _ in range(len(first_named_tuple))]
    for k, v in experience.items():
        for i, _v in enumerate(v):
            _args[i][k] = _v

    return _cls(*_args)


@dataclass
class ExperiencePipeline:
    """Dataclass for ExperiencePipeline.

    An ExperiencePipeline handles a sequence of transformations designed for
    single agent and non-vectorized environment, ie:
    ```
    observation = Array(T, 4,) # transforms designed for this
    observation = {"a": Array(T, 4,), "b": Array(T, 4,)} # not for this
    observation = Array(T, E, 4) # not for this
    ```
    with T the number of steps and E the number of environments.

    `ExperiencePipeline.run` automatically broadcast the sequence of transforms to
    multi-agent and vectorized environments by treating each agent of each environment
    separately, then merging the results.

    In a sequence of transforms, the output of the previous item should be of the
    same type as the expected input of the next item. **No error will be raised**.

    If no transforms are provided, the pipeline will simply merge the agents and the environments
    without applying any transformation.

    Attributes:
        transforms: A list of ExperienceTransform to execute sequentially
        vectorized: A boolean that indicates if the environment is vectorized
        parallel: A boolean that indicates if the environment is multi-agent
    """

    transforms: list[ExperienceTransform]

    vectorized: bool = True
    parallel: bool = False

    def run_single_pipe(
        self, state: AgentPyTree, key: PRNGKeyArray, experience: NamedTuple
    ) -> NamedTuple:
        """Runs the transforms sequentially for a single agent in a single environment."""
        rng = PRNGSequence(key)
        for t in self.transforms:
            experience = t(state, next(rng), experience)
        return experience

    def run(
        self, state: AgentPyTree, key: PRNGKeyArray, experience: NamedTuple
    ) -> NamedTuple:
        """Sequentially runs the experience transforms.

        Args:
            state: An AgentPyTree that contains the agent's state
            key: A PRNGKeyArray for reproducibility
            experience: A NamedTuple of the same type as the first transform's input

        Returns:
            A processed NamedTuple.
        """
        if self.parallel and self.vectorized:
            return self.run_parallel_vectorized(state, key, experience)

        if self.parallel:
            return self.run_parallel(state, key, experience)

        if self.vectorized:
            return self.run_vectorized(state, key, experience)

        return self.run_single_pipe(state, key, experience)

    def run_parallel(
        self, state: AgentPyTree, key: PRNGKeyArray, experience: NamedTuple
    ) -> NamedTuple:
        """Runs a single pipe in parallel.

        Experiences should be provided as a tuple of dictionaries, eg:
        ```
        experience = Foo(
            a={"a": array_a(shape=(5, 3)), "b": array_b(shape=(5, 3))},
            b={"a": Array_A(shape=(5,)), "b": Array_B(shape=(5,))}
        )
        ```

        The output is a tuple of Arrays where keys are merged, eg:
        ```
        output = Foo(
            a=array_a / array_b (shape=(10, 3)),
            b=Array_A / Array_b (shape=(10,))
        )
        ```

        Args:
            state: An AgentPyTree state
            key: A PRNGKeyArray for reproducibility
            experience: A NamedTuple of dictionaries of Arrays
        Returns:
            A processed NamedTuple where keys are merged by concatenating arrays.
        """
        _run_single_pipe = lambda k, e: self.run_single_pipe(state, k, e)

        keys = {}
        for agent in experience[0].keys():
            key, keys[agent] = jax.random.split(key, 2)

        dict_experience: dict[str, NamedTuple] = tuple_to_dict(experience)
        processed_experience = jax.tree_map(_run_single_pipe, keys, dict_experience)

        return jax.tree_map(
            lambda *x: stack_and_merge_n_first_dims(x, 2),
            *zip(processed_experience.values()),
        )[0]

    def run_vectorized(
        self, state: AgentPyTree, key: PRNGKeyArray, experience: NamedTuple
    ) -> NamedTuple:
        """Runs a single pipe in vectorized.

        Experiences should be provided as a tuple of Arrays with at least 2 dimensions,
        where the first two dimensions should be equal. Eg:
        ```
        experience = Foo(a=Array(shape=(5, 10, 3)), b=Array(shape=(5, 10)))
        ```

        The output is a tuple of Arrays where the first 2 dimensions are concatenated, eg:
        ```
        output = Foo(a=Array(shape=(50, 3)), b=Array(shape=(50,)))
        ```

        Args:
            state: An AgentPyTree state
            key: A PRNGKeyArray for reproducibility
            experience: A NamedTuple of Arrays with at least the two first dimensions identical
        Returns:
            A processed NamedTuple where the first two dimensions are concatenated.
        """
        _run_single_pipe = lambda k, e: self.run_single_pipe(state, k, e)
        run_vectorized_pipe = jax.vmap(_run_single_pipe, in_axes=1, out_axes=1)

        keys = jax.random.split(key, experience[0].shape[1]).T

        processed_experience = run_vectorized_pipe(keys, experience)

        return jax.tree_map(lambda x: merge_n_first_dims(x, 2), processed_experience)

    def run_parallel_vectorized(
        self, state: AgentPyTree, key: PRNGKeyArray, experience: NamedTuple
    ) -> NamedTuple:
        """Runs a single pipe in parallel and vectorized.

        Experiences should be provided as a tuple of dictionaries, where values are
        Arrays with at least 2 dimensions, where the first two dimensions should be equal.
        Eg:
        ```
        experience = Foo(
            a={"a": array_a(shape=(5, 10, 3)), "b": array_b(shape=(5, 10, 3))},
            b={"a": Array_A(shape=(5, 10)), "b": Array_B(shape=(5, 10))}
        )
        ```

        The output is a tuple of Arrays where keys are merged, eg:
        ```
        output = Foo(
            a=array_a / array_b (shape=(100, 3)),
            b=Array_A / Array_b (shape=(100))
        )
        ```

        Args:
            state: An AgentPyTree state
            key: A PRNGKeyArray for reproducibility
            experience: A NamedTuple of dictionaries of Arrays with at least
                the two first dimensions identical
        Returns:
            A processed NamedTuple where keys are the first two dimensions are merged.
        """
        _run_single_pipe = lambda k, e: self.run_single_pipe(state, k, e)
        run_vectorized_pipe = jax.vmap(_run_single_pipe, in_axes=1, out_axes=1)

        keys = {}
        for agent, value in experience[0].items():
            # vectorized: shape=(T, n_envs, ...)
            _keys = jax.random.split(key, value.shape[1] + 1)
            key, keys[agent] = _keys[0], _keys[1:].T

        dict_experience: dict[str, NamedTuple] = tuple_to_dict(experience)
        processed_experience = jax.tree_map(run_vectorized_pipe, keys, dict_experience)

        return jax.tree_map(
            lambda *x: stack_and_merge_n_first_dims(x, 3),
            *zip(processed_experience.values()),
        )[0]
