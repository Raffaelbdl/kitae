import jax.random as jrd


class Seeded:
    def __init__(self, seed: int):
        self.seed = seed
        self.key = jrd.PRNGKey(seed)

    def nextkey(self):
        self.key, _k = jrd.split(self.key)
        return _k
