import flax.linen as nn
import jax.numpy as jnp

from kitae.modules.modules import parallel_copy


def test_parallel_copy():
    class DummyModule(nn.Module):
        @nn.compact
        def __call__(self, x1, x2):
            return x1 + x2

    module = DummyModule()
    parallel_module = parallel_copy(module, 5)

    x1, x2 = jnp.ones((5,)), jnp.ones((5,))
    m1 = module.apply({}, x1, x2)
    assert jnp.array_equal(m1, x1 + x2)

    m1, m2, m3, m4, m5 = parallel_module.apply({}, x1, x2)
    assert jnp.array_equal(m1, x1 + x2)
    assert jnp.array_equal(m2, x1 + x2)
    assert jnp.array_equal(m3, x1 + x2)
    assert jnp.array_equal(m4, x1 + x2)
    assert jnp.array_equal(m5, x1 + x2)
