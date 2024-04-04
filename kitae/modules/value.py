import flax.linen as nn
import jax


class ValueOutput(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Dense(
            features=1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
